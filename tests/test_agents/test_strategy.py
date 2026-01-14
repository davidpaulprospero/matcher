"""
Tests for HealingStrategy and related classes.
"""

import pytest
from unittest.mock import Mock

from src.agents.strategy import (
    HealingStrategy,
    HealingMode,
    HealingMetrics,
    ConfigSnapshot,
)


class TestHealingMode:
    """Tests for HealingMode enum."""

    def test_mode_values(self):
        """Test all mode values exist."""
        assert HealingMode.AGGRESSIVE.value == "aggressive"
        assert HealingMode.CONSERVATIVE.value == "conservative"
        assert HealingMode.INTERACTIVE.value == "interactive"
        assert HealingMode.MINIMAL.value == "minimal"


class TestHealingStrategy:
    """Tests for HealingStrategy configuration."""

    def test_default_strategy(self):
        """Test default strategy values."""
        strategy = HealingStrategy()

        assert strategy.mode == HealingMode.CONSERVATIVE
        assert strategy.max_attempts_per_stage == 3
        assert strategy.max_total_heals == 20
        assert strategy.run_preflight is True
        assert strategy.enable_rollback is True

    def test_aggressive_factory(self):
        """Test aggressive() factory method."""
        strategy = HealingStrategy.aggressive()

        assert strategy.mode == HealingMode.AGGRESSIVE
        assert strategy.max_attempts_per_stage == 5
        assert strategy.max_total_heals == 50
        assert strategy.heal_delay == 1.0

    def test_conservative_factory(self):
        """Test conservative() factory method."""
        strategy = HealingStrategy.conservative()

        assert strategy.mode == HealingMode.CONSERVATIVE
        assert strategy.max_attempts_per_stage == 3
        assert strategy.max_total_heals == 20
        assert strategy.heal_delay == 2.0

    def test_interactive_factory(self):
        """Test interactive() factory method."""
        strategy = HealingStrategy.interactive()

        assert strategy.mode == HealingMode.INTERACTIVE
        assert strategy.auto_fix_preflight is False  # Asks user first

    def test_minimal_factory(self):
        """Test minimal() factory method."""
        strategy = HealingStrategy.minimal()

        assert strategy.mode == HealingMode.MINIMAL
        assert strategy.max_attempts_per_stage == 1
        assert strategy.max_total_heals == 5
        assert strategy.enable_rollback is False

    def test_healer_priority_default(self):
        """Test default healer priority order."""
        strategy = HealingStrategy()

        assert strategy.healer_priority[0] == "checkpoint-healer"
        assert "otio-healer" in strategy.healer_priority

    def test_always_escalate_patterns(self):
        """Test always_escalate contains critical patterns."""
        strategy = HealingStrategy()

        assert "permission denied" in strategy.always_escalate
        assert "api key invalid" in strategy.always_escalate

    def test_protected_config_keys(self):
        """Test protected_config_keys contains sensitive keys."""
        strategy = HealingStrategy()

        assert "api_key" in strategy.protected_config_keys
        assert "password" in strategy.protected_config_keys
        assert "token" in strategy.protected_config_keys

    def test_skip_healers(self):
        """Test skip_healers set."""
        strategy = HealingStrategy(skip_healers={"disk-healer"})

        assert "disk-healer" in strategy.skip_healers


class TestHealingMetrics:
    """Tests for HealingMetrics tracking."""

    def test_default_metrics(self):
        """Test default metric values."""
        metrics = HealingMetrics()

        assert metrics.total_heals == 0
        assert metrics.successful_heals == 0
        assert metrics.failed_heals == 0
        assert metrics.time_spent_healing == 0.0

    def test_record_heal_success(self):
        """Test recording successful heal."""
        metrics = HealingMetrics()

        metrics.record_heal("otio-healer", "OUTPUT", success=True)

        assert metrics.total_heals == 1
        assert metrics.successful_heals == 1
        assert metrics.failed_heals == 0
        assert metrics.heals_by_stage["OUTPUT"] == 1
        assert metrics.heals_by_healer["otio-healer"] == 1

    def test_record_heal_failure(self):
        """Test recording failed heal."""
        metrics = HealingMetrics()

        metrics.record_heal("api-healer", "DOWNLOAD", success=False)

        assert metrics.total_heals == 1
        assert metrics.successful_heals == 0
        assert metrics.failed_heals == 1

    def test_record_multiple_heals(self):
        """Test recording multiple heals."""
        metrics = HealingMetrics()

        metrics.record_heal("otio-healer", "OUTPUT", success=True)
        metrics.record_heal("otio-healer", "OUTPUT", success=True)
        metrics.record_heal("api-healer", "DOWNLOAD", success=False)

        assert metrics.total_heals == 3
        assert metrics.successful_heals == 2
        assert metrics.failed_heals == 1
        assert metrics.heals_by_stage["OUTPUT"] == 2
        assert metrics.heals_by_healer["otio-healer"] == 2

    def test_summary(self):
        """Test summary generation."""
        metrics = HealingMetrics()
        metrics.record_heal("otio-healer", "OUTPUT", success=True)
        metrics.preflight_issues_found = 3
        metrics.preflight_issues_fixed = 2

        summary = metrics.summary()

        assert "Total heals: 1" in summary
        assert "otio-healer" in summary
        assert "Preflight: 2/3" in summary


class TestConfigSnapshot:
    """Tests for ConfigSnapshot and rollback."""

    def test_snapshot_creation(self):
        """Test creating a config snapshot."""
        import time

        snapshot = ConfigSnapshot(
            stage_name="OUTPUT",
            timestamp=time.time(),
            config_values={"output.gap_mode": "scale"}
        )

        assert snapshot.stage_name == "OUTPUT"
        assert "output.gap_mode" in snapshot.config_values

    def test_restore_simple_config(self):
        """Test restoring simple config values."""
        config = Mock()
        config.output = Mock()
        config.output.gap_mode = "none"

        snapshot = ConfigSnapshot(
            stage_name="OUTPUT",
            timestamp=0,
            config_values={"output.gap_mode": "scale"}
        )

        success = snapshot.restore(config)

        assert success is True
        assert config.output.gap_mode == "scale"

    def test_restore_multiple_values(self):
        """Test restoring multiple config values."""
        config = Mock()
        config.output = Mock()
        config.output.gap_mode = "none"
        config.output.frame_rate = 24.0

        snapshot = ConfigSnapshot(
            stage_name="OUTPUT",
            timestamp=0,
            config_values={
                "output.gap_mode": "scale",
                "output.frame_rate": 30.0
            }
        )

        snapshot.restore(config)

        assert config.output.gap_mode == "scale"
        assert config.output.frame_rate == 30.0

    def test_restore_handles_missing_section(self):
        """Test restore handles missing config sections gracefully."""
        config = Mock()
        config.nonexistent = None

        snapshot = ConfigSnapshot(
            stage_name="TEST",
            timestamp=0,
            config_values={"nonexistent.value": "test"}
        )

        # Should not raise
        success = snapshot.restore(config)
        # Returns False since nothing was restored
        assert success is False
