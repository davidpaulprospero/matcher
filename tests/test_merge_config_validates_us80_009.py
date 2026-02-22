"""
Tests for US-80-009: merge_config() re-runs validate() after applying overrides.

Verifies that invalid overrides produce WARNING logs and valid overrides do not.
"""

import logging
import pytest

from src.config.base import Config
from src.cli.config_utils import merge_config


@pytest.mark.fast
class TestMergeConfigValidatesAfterMerge:
    """merge_config() calls config.validate() after overrides and logs warnings."""

    def test_invalid_min_confidence_logs_warning(self, caplog):
        """Override matching.min_confidence=2.0 produces a validation warning."""
        config = Config()
        overrides = {"matching": {"min_confidence": 2.0}}

        with caplog.at_level(logging.WARNING, logger="src.cli.config_utils"):
            result = merge_config(config, overrides)

        # Should have logged a warning about min_confidence out of range
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("min_confidence" in msg for msg in warning_messages), (
            f"Expected a warning about min_confidence, got: {warning_messages}"
        )

    def test_invalid_override_does_not_raise(self):
        """merge_config() does NOT raise on validation warnings, only logs them."""
        config = Config()
        overrides = {"matching": {"min_confidence": 2.0}}

        # Should not raise
        result = merge_config(config, overrides)
        assert result is config

    def test_valid_overrides_no_range_validation_warnings(self, caplog):
        """Valid overrides produce no range/constraint validation warnings."""
        config = Config()
        overrides = {"matching": {"min_confidence": 0.8}}

        with caplog.at_level(logging.WARNING, logger="src.cli.config_utils"):
            merge_config(config, overrides)

        # Filter to only range/constraint warnings (not API key warnings which
        # fire because test Config has no keys set)
        range_warnings = [
            r.message for r in caplog.records
            if r.levelno == logging.WARNING
            and "Post-merge validation" in r.message
            and "min_confidence" in r.message
        ]
        assert range_warnings == [], (
            f"Expected no min_confidence warnings for valid value, got: {range_warnings}"
        )

    def test_returns_config_with_override_applied(self):
        """merge_config() still applies the override value."""
        config = Config()
        overrides = {"matching": {"min_confidence": 0.9}}

        result = merge_config(config, overrides)

        assert result.matching.min_confidence == 0.9
