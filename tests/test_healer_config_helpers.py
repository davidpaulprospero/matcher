"""Tests for get_config_value and set_config_value helpers in agents.base."""

import pytest
from dataclasses import dataclass

from src.agents.base import get_config_value, set_config_value


@dataclass
class FakeConfig:
    """Dataclass config for testing."""
    timeout: int = 30
    provider: str = "gemini"


class TestGetConfigValue:
    """Tests for get_config_value helper."""

    def test_dict_config(self):
        """get_config_value reads from a dict config."""
        config = {"timeout": 60, "provider": "anthropic"}
        assert get_config_value(config, "timeout", 30) == 60
        assert get_config_value(config, "provider", "gemini") == "anthropic"

    def test_dict_config_missing_key_returns_default(self):
        """get_config_value returns default for missing dict key."""
        config = {"timeout": 60}
        assert get_config_value(config, "missing_field", "fallback") == "fallback"

    def test_dataclass_config(self):
        """get_config_value reads from a dataclass config."""
        config = FakeConfig(timeout=120, provider="ollama")
        assert get_config_value(config, "timeout", 30) == 120
        assert get_config_value(config, "provider", "gemini") == "ollama"

    def test_dataclass_config_missing_attr_returns_default(self):
        """get_config_value returns default for missing dataclass attr."""
        config = FakeConfig()
        assert get_config_value(config, "nonexistent", "default_val") == "default_val"

    def test_none_config_returns_default(self):
        """get_config_value returns default when config is None."""
        assert get_config_value(None, "timeout", 30) == 30
        assert get_config_value(None, "anything", "fallback") == "fallback"

    def test_none_config_default_is_none(self):
        """get_config_value returns None when config is None and no default."""
        assert get_config_value(None, "field") is None

    def test_dict_with_none_value(self):
        """get_config_value returns None (not default) when key exists with None value."""
        config = {"timeout": None}
        assert get_config_value(config, "timeout", 30) is None

    def test_dataclass_with_falsy_value(self):
        """get_config_value returns falsy values correctly (0, empty string)."""
        config = FakeConfig(timeout=0)
        assert get_config_value(config, "timeout", 30) == 0


class TestSetConfigValue:
    """Tests for set_config_value helper."""

    def test_set_on_dict(self):
        """set_config_value sets value on dict."""
        config = {"timeout": 30}
        result = set_config_value(config, "timeout", 60)
        assert result is True
        assert config["timeout"] == 60

    def test_set_new_key_on_dict(self):
        """set_config_value adds new key to dict."""
        config = {}
        result = set_config_value(config, "new_field", "value")
        assert result is True
        assert config["new_field"] == "value"

    def test_set_on_dataclass(self):
        """set_config_value sets value on dataclass."""
        config = FakeConfig(timeout=30)
        result = set_config_value(config, "timeout", 120)
        assert result is True
        assert config.timeout == 120

    def test_set_on_none_returns_false(self):
        """set_config_value returns False when config is None."""
        result = set_config_value(None, "timeout", 60)
        assert result is False
