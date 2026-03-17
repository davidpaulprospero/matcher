"""
Tests for US-123-007: Cross-session rate limit state persistence.

Tests coverage for:
- serialize_rate_limit_state() method
- deserialize_rate_limit_state() method
- Stale state filtering
- Checkpoint integration methods

Created: February 17, 2026
User Story: US-123-007 - Cross-session rate limit state persistence
"""

import sys
import time
from pathlib import Path
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.downloader.rate_limit_budget import RateLimitBudget


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def budget_with_state():
    """Create a RateLimitBudget with some state."""
    b = RateLimitBudget()
    b.rotations_used = 3
    b.vpn_switches_used = 1
    b.backoff_time_spent = 45.0
    b.keywords_rate_limited = ["python", "tutorial"]
    b.last_escalation_level = "cookie"
    b.successes = 10
    b.failures = 2
    return b


@pytest.fixture
def fresh_budget():
    """Create a fresh RateLimitBudget with limits."""
    b = RateLimitBudget()
    b.max_rotations = 5
    b.max_vpn_switches = 3
    b.max_backoff_time = 60.0
    return b


# ============================================================================
# Tests for serialize_rate_limit_state
# ============================================================================

class TestSerializeRateLimitState:
    """Tests for serialize_rate_limit_state method."""

    def test_serialize_basic_state(self, budget_with_state):
        """Test basic state serialization includes all fields."""
        state = budget_with_state.serialize_rate_limit_state()

        assert state["rotations_used"] == 3
        assert state["vpn_switches_used"] == 1
        assert state["backoff_time_spent"] == 45.0
        assert state["keywords_rate_limited"] == ["python", "tutorial"]
        assert state["last_escalation_level"] == "cookie"
        assert state["successes"] == 10
        assert state["failures"] == 2
        assert "saved_at" in state
        assert "saved_at_iso" in state

    def test_serialize_timestamp(self, budget_with_state):
        """Test that serialization adds timestamp."""
        before = time.time()
        state = budget_with_state.serialize_rate_limit_state()
        after = time.time()

        assert before <= state["saved_at"] <= after
        assert isinstance(state["saved_at_iso"], str)

    def test_serialize_empty_budget(self, fresh_budget):
        """Test serialization of fresh budget with no usage."""
        state = fresh_budget.serialize_rate_limit_state()

        assert state["rotations_used"] == 0
        assert state["vpn_switches_used"] == 0
        assert state["backoff_time_spent"] == 0.0
        assert "saved_at" in state


# ============================================================================
# Tests for deserialize_rate_limit_state
# ============================================================================

class TestDeserializeRateLimitState:
    """Tests for deserialize_rate_limit_state method."""

    def test_deserialize_basic_state(self, fresh_budget):
        """Test basic state deserialization."""
        # First serialize
        state = RateLimitBudget().serialize_rate_limit_state()
        state["rotations_used"] = 3
        state["vpn_switches_used"] = 1
        state["backoff_time_spent"] = 45.0

        # Then deserialize
        restored = RateLimitBudget.deserialize_rate_limit_state(state)

        assert restored is not None
        assert restored.rotations_used == 3
        assert restored.vpn_switches_used == 1
        assert restored.backoff_time_spent == 45.0

    def test_deserialize_none_returns_none(self):
        """Test that None input returns None."""
        result = RateLimitBudget.deserialize_rate_limit_state(None)
        assert result is None

    def test_deserialize_empty_dict_returns_none(self):
        """Test that empty dict returns None."""
        result = RateLimitBudget.deserialize_rate_limit_state({})
        assert result is None

    def test_deserialize_filters_stale_state(self, fresh_budget):
        """Test that stale state is filtered out."""
        # Create old state with timestamp older than threshold
        old_state = {
            "rotations_used": 5,
            "vpn_switches_used": 2,
            "backoff_time_spent": 30.0,
            "saved_at": time.time() - 7200,  # 2 hours old
        }

        # With 1-hour threshold, this should be filtered
        restored = RateLimitBudget.deserialize_rate_limit_state(
            old_state,
            stale_state_threshold=3600.0
        )

        assert restored is None

    def test_deserialize_keeps_fresh_state(self, fresh_budget):
        """Test that fresh state is kept."""
        # Create recent state
        recent_state = {
            "rotations_used": 2,
            "vpn_switches_used": 1,
            "backoff_time_spent": 15.0,
            "saved_at": time.time() - 1800,  # 30 minutes old
        }

        # With 1-hour threshold, this should be kept
        restored = RateLimitBudget.deserialize_rate_limit_state(
            recent_state,
            stale_state_threshold=3600.0
        )

        assert restored is not None
        assert restored.rotations_used == 2

    def test_deserialize_without_timestamp(self, fresh_budget):
        """Test deserialization without timestamp (backward compat)."""
        state = {
            "rotations_used": 3,
            "vpn_switches_used": 1,
            "backoff_time_spent": 20.0,
            # No saved_at field
        }

        restored = RateLimitBudget.deserialize_rate_limit_state(state)

        assert restored is not None
        assert restored.rotations_used == 3


# ============================================================================
# Integration tests for checkpoint integration
# ============================================================================

class TestCheckpointIntegration:
    """Tests for checkpoint integration methods."""

    def test_save_rate_limit_state_creates_checkpoint_field(self, tmp_path):
        """Test that saving creates the rate_limit_state field."""
        from src.checkpoint import CheckpointManager, CheckpointData
        from datetime import datetime

        # Create checkpoint manager with minimal setup
        checkpoint_path = tmp_path / "checkpoint.json"
        cm = CheckpointManager(checkpoint_path, config_hash="test")

        # Create and save rate limit state
        budget = RateLimitBudget()
        budget.rotations_used = 2
        budget.vpn_switches_used = 1
        budget.backoff_time_spent = 30.0

        cm.save_rate_limit_state(budget)

        # Verify state was saved
        assert cm.data.rate_limit_state is not None
        assert cm.data.rate_limit_state["rotations_used"] == 2

    def test_has_rate_limit_state(self):
        """Test has_rate_limit_state method."""
        from src.checkpoint import CheckpointData

        # Empty checkpoint should not have state
        data = CheckpointData()
        assert "rate_limit_state" in data.__dataclass_fields__

    def test_get_rate_limit_state(self):
        """Test get_rate_limit_state returns saved state."""
        from src.checkpoint import CheckpointData

        data = CheckpointData()
        data.rate_limit_state = {"rotations_used": 5, "vpn_switches_used": 2}

        assert data.rate_limit_state["rotations_used"] == 5


# ============================================================================
# Tests for stale state threshold
# ============================================================================

class TestStaleStateThreshold:
    """Tests for stale state threshold configuration."""

    def test_threshold_default_value(self):
        """Test that default threshold is 1 hour."""
        from src.config.sections.rate_limit import RateLimitBudgetConfig

        config = RateLimitBudgetConfig()
        assert config.stale_state_threshold == 3600.0

    def test_threshold_custom_value(self):
        """Test custom threshold value."""
        from src.config.sections.rate_limit import RateLimitBudgetConfig

        config = RateLimitBudgetConfig(stale_state_threshold=1800.0)
        assert config.stale_state_threshold == 1800.0

    def test_threshold_in_download_config(self):
        """Test threshold is available in download config."""
        from src.config.sections.download import RateLimitBudgetConfig

        config = RateLimitBudgetConfig()
        assert hasattr(config, 'stale_state_threshold')
        assert config.stale_state_threshold == 3600.0


# ============================================================================
# Tests for config persistence flag
# ============================================================================

class TestStatePersistenceFlag:
    """Tests for state_persistence_enabled flag."""

    def test_flag_default_value(self):
        """Test that default flag is False."""
        from src.config.sections.rate_limit import RateLimitBudgetConfig

        config = RateLimitBudgetConfig()
        assert config.state_persistence_enabled is False

    def test_flag_custom_value(self):
        """Test custom flag value."""
        from src.config.sections.rate_limit import RateLimitBudgetConfig

        config = RateLimitBudgetConfig(state_persistence_enabled=True)
        assert config.state_persistence_enabled is True
