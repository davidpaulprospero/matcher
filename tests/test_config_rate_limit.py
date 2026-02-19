"""
Unit tests for config/sections/rate_limit module.

Tests RateLimitConfig dataclass from src/config/sections/rate_limit.py.
"""

import pytest
import sys
import yaml
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.sections.rate_limit import RateLimitConfig, RateLimitBudgetConfig


class TestRateLimitConfigDefaults:
    """Test RateLimitConfig initialization with default values."""

    @pytest.mark.fast
    def test_default_values(self):
        """Test initialization with default values."""
        config = RateLimitConfig()

        assert config.slots_per_second == 0.5
        assert config.burst_size == 3
        assert config.jitter_factor == 0.2
        assert config.max_backoff_seconds == 60.0

    @pytest.mark.fast
    def test_all_defaults_match_docstring(self):
        """Test that defaults match documented values."""
        config = RateLimitConfig()

        # Verify all documented defaults
        assert config.slots_per_second == 0.5  # 1 request every 2 seconds
        assert config.burst_size == 3
        assert config.jitter_factor == 0.2  # 0.0-1.0 range
        assert config.max_backoff_seconds == 60.0


class TestRateLimitConfigCustom:
    """Test RateLimitConfig with custom values."""

    @pytest.mark.fast
    def test_custom_values(self):
        """Test initialization with custom values."""
        config = RateLimitConfig(
            slots_per_second=1.0,
            burst_size=5,
            jitter_factor=0.5,
            max_backoff_seconds=120.0
        )

        assert config.slots_per_second == 1.0
        assert config.burst_size == 5
        assert config.jitter_factor == 0.5
        assert config.max_backoff_seconds == 120.0

    @pytest.mark.fast
    def test_aggressive_custom_values(self):
        """Test with aggressive rate limiting settings."""
        config = RateLimitConfig(
            slots_per_second=2.0,  # Very aggressive
            burst_size=10,
            jitter_factor=0.0,  # No jitter
            max_backoff_seconds=30.0
        )

        assert config.slots_per_second == 2.0
        assert config.burst_size == 10
        assert config.jitter_factor == 0.0
        assert config.max_backoff_seconds == 30.0

    @pytest.mark.fast
    def test_conservative_custom_values(self):
        """Test with conservative rate limiting settings."""
        config = RateLimitConfig(
            slots_per_second=0.1,  # Very slow
            burst_size=1,
            jitter_factor=1.0,  # Maximum jitter
            max_backoff_seconds=300.0  # 5 minutes
        )

        assert config.slots_per_second == 0.1
        assert config.burst_size == 1
        assert config.jitter_factor == 1.0
        assert config.max_backoff_seconds == 300.0


class TestRateLimitConfigBoundaries:
    """Test boundary conditions and edge cases."""

    @pytest.mark.fast
    def test_zero_slots_per_second(self):
        """Test with zero slots per second (no requests)."""
        config = RateLimitConfig(slots_per_second=0.0)
        assert config.slots_per_second == 0.0

    @pytest.mark.fast
    def test_zero_burst_size(self):
        """Test with zero burst size."""
        config = RateLimitConfig(burst_size=0)
        assert config.burst_size == 0

    @pytest.mark.fast
    def test_zero_jitter_factor(self):
        """Test with zero jitter (deterministic timing)."""
        config = RateLimitConfig(jitter_factor=0.0)
        assert config.jitter_factor == 0.0

    @pytest.mark.fast
    def test_max_jitter_factor(self):
        """Test with max jitter (maximum randomization)."""
        config = RateLimitConfig(jitter_factor=1.0)
        assert config.jitter_factor == 1.0

    @pytest.mark.fast
    def test_zero_max_backoff(self):
        """Test with zero max backoff (immediate escalation)."""
        config = RateLimitConfig(max_backoff_seconds=0.0)
        assert config.max_backoff_seconds == 0.0

    @pytest.mark.fast
    def test_large_values(self):
        """Test with large values."""
        config = RateLimitConfig(
            slots_per_second=100.0,
            burst_size=1000,
            jitter_factor=0.99,
            max_backoff_seconds=3600.0  # 1 hour
        )

        assert config.slots_per_second == 100.0
        assert config.burst_size == 1000
        assert config.jitter_factor == 0.99
        assert config.max_backoff_seconds == 3600.0


class TestRateLimitConfigTypes:
    """Test type validation and coercion."""

    @pytest.mark.fast
    def test_integer_burst_size(self):
        """Test that burst_size accepts integers."""
        config = RateLimitConfig(burst_size=3)
        assert isinstance(config.burst_size, int)
        assert config.burst_size == 3

    @pytest.mark.fast
    def test_float_slots_per_second(self):
        """Test that slots_per_second accepts floats."""
        config = RateLimitConfig(slots_per_second=0.5)
        assert isinstance(config.slots_per_second, float)
        assert config.slots_per_second == 0.5


class TestRateLimitConfigSerialization:
    """Test dataclass serialization/deserialization."""

    @pytest.mark.fast
    def test_to_dict(self):
        """Test RateLimitConfig can be converted to dict."""
        config = RateLimitConfig(
            slots_per_second=0.75,
            burst_size=4,
            jitter_factor=0.3,
            max_backoff_seconds=90.0
        )

        result = {
            'slots_per_second': config.slots_per_second,
            'burst_size': config.burst_size,
            'jitter_factor': config.jitter_factor,
            'max_backoff_seconds': config.max_backoff_seconds
        }

        assert result['slots_per_second'] == 0.75
        assert result['burst_size'] == 4
        assert result['jitter_factor'] == 0.3
        assert result['max_backoff_seconds'] == 90.0

    @pytest.mark.fast
    def test_roundtrip_via_dict(self):
        """Test RateLimitConfig survives dict roundtrip."""
        original = RateLimitConfig(
            slots_per_second=0.8,
            burst_size=5,
            jitter_factor=0.25,
            max_backoff_seconds=45.0
        )

        # Serialize to dict
        data = {
            'slots_per_second': original.slots_per_second,
            'burst_size': original.burst_size,
            'jitter_factor': original.jitter_factor,
            'max_backoff_seconds': original.max_backoff_seconds
        }

        # Deserialize back to dataclass
        restored = RateLimitConfig(**data)

        assert restored.slots_per_second == 0.8
        assert restored.burst_size == 5
        assert restored.jitter_factor == 0.25
        assert restored.max_backoff_seconds == 45.0

    @pytest.mark.fast
    def test_yaml_roundtrip(self):
        """Test RateLimitConfig survives YAML roundtrip."""
        original = RateLimitConfig(
            slots_per_second=0.6,
            burst_size=2,
            jitter_factor=0.15,
            max_backoff_seconds=30.0
        )

        # Serialize to YAML
        yaml_str = yaml.dump({
            'slots_per_second': original.slots_per_second,
            'burst_size': original.burst_size,
            'jitter_factor': original.jitter_factor,
            'max_backoff_seconds': original.max_backoff_seconds
        })

        # Deserialize from YAML
        data = yaml.safe_load(yaml_str)
        restored = RateLimitConfig(**data)

        assert restored.slots_per_second == 0.6
        assert restored.burst_size == 2
        assert restored.jitter_factor == 0.15
        assert restored.max_backoff_seconds == 30.0


class TestRateLimitConfigBackoff:
    """Test backoff configuration (max_backoff_seconds field)."""

    @pytest.mark.fast
    def test_backoff_multiplier_defaults(self):
        """Test backoff defaults."""
        config = RateLimitConfig()

        # max_backoff_seconds is the backoff ceiling
        assert config.max_backoff_seconds == 60.0

    @pytest.mark.fast
    def test_custom_backoff(self):
        """Test custom backoff threshold."""
        config = RateLimitConfig(max_backoff_seconds=180.0)  # 3 minutes

        assert config.max_backoff_seconds == 180.0

    @pytest.mark.fast
    def test_backoff_edge_cases(self):
        """Test edge cases for backoff configuration."""
        # Very short backoff
        config = RateLimitConfig(max_backoff_seconds=1.0)
        assert config.max_backoff_seconds == 1.0

        # Very long backoff (1 day)
        config = RateLimitConfig(max_backoff_seconds=86400.0)
        assert config.max_backoff_seconds == 86400.0


class TestRateLimitConfigImmutability:
    """Test that dataclass behaves correctly (no frozen=True, so mutable)."""

    @pytest.mark.fast
    def test_fields_are_modifiable(self):
        """Test that fields can be modified after creation."""
        config = RateLimitConfig()
        config.slots_per_second = 1.0
        config.burst_size = 10

        assert config.slots_per_second == 1.0
        assert config.burst_size == 10


class TestRateLimitConfigEquality:
    """Test dataclass equality and comparison."""

    @pytest.mark.fast
    def test_equal_configs(self):
        """Test that identical configs are equal."""
        config1 = RateLimitConfig(slots_per_second=0.5, burst_size=3)
        config2 = RateLimitConfig(slots_per_second=0.5, burst_size=3)

        assert config1 == config2

    @pytest.mark.fast
    def test_unequal_configs(self):
        """Test that different configs are not equal."""
        config1 = RateLimitConfig(slots_per_second=0.5)
        config2 = RateLimitConfig(slots_per_second=1.0)

        assert config1 != config2


class TestRateLimitBudgetConfig:
    """Test RateLimitBudgetConfig dataclass."""

    @pytest.mark.fast
    def test_budget_defaults(self):
        """Test RateLimitBudgetConfig initialization with default values."""
        budget = RateLimitBudgetConfig()

        assert budget.max_rotations == 10
        assert budget.max_backoff_time == 600.0
        assert budget.max_vpn_switches == 3
        assert budget.tier_max_attempts == {}

    @pytest.mark.fast
    def test_budget_custom_values(self):
        """Test RateLimitBudgetConfig with custom values."""
        budget = RateLimitBudgetConfig(
            max_rotations=5,
            max_backoff_time=300.0,
            max_vpn_switches=2
        )

        assert budget.max_rotations == 5
        assert budget.max_backoff_time == 300.0
        assert budget.max_vpn_switches == 2

    @pytest.mark.fast
    def test_budget_tier_max_attempts(self):
        """Test RateLimitBudgetConfig with tier_max_attempts."""
        tier_attempts = {'tier1': 3, 'tier2': 3, 'tier3': 2, 'tier4': 1}
        budget = RateLimitBudgetConfig(tier_max_attempts=tier_attempts)

        assert budget.tier_max_attempts == tier_attempts


class TestRateLimitConfigPostInit:
    """Test RateLimitConfig __post_init__ dict conversion."""

    @pytest.mark.fast
    def test_post_init_converts_dict_budget(self):
        """Test __post_init__ converts dict budget to RateLimitBudgetConfig."""
        # Create config with budget as dict (like YAML would provide)
        config = RateLimitConfig(
            slots_per_second=1.0,
            budget={
                'max_rotations': 5,
                'max_backoff_time': 300.0,
                'max_vpn_switches': 2
            }
        )

        # Should convert dict to RateLimitBudgetConfig
        assert isinstance(config.budget, RateLimitBudgetConfig)
        assert config.budget.max_rotations == 5
        assert config.budget.max_backoff_time == 300.0
        assert config.budget.max_vpn_switches == 2

    @pytest.mark.fast
    def test_post_init_handles_none_budget(self):
        """Test __post_init__ handles None budget dict."""
        config = RateLimitConfig(budget=None)

        # Should create default budget
        assert isinstance(config.budget, RateLimitBudgetConfig)
        assert config.budget.max_rotations == 10

    @pytest.mark.fast
    def test_post_init_handles_empty_dict(self):
        """Test __post_init__ handles empty dict budget."""
        config = RateLimitConfig(budget={})

        # Should create budget with defaults
        assert isinstance(config.budget, RateLimitBudgetConfig)
        assert config.budget.max_rotations == 10

    @pytest.mark.fast
    def test_nested_budget_with_tier_max_attempts(self):
        """Test nested budget with tier_max_attempts dict."""
        config = RateLimitConfig(
            budget={
                'max_rotations': 8,
                'tier_max_attempts': {'tier1': 5, 'tier2': 5}
            }
        )

        assert isinstance(config.budget, RateLimitBudgetConfig)
        assert config.budget.max_rotations == 8
        assert config.budget.tier_max_attempts == {'tier1': 5, 'tier2': 5}


class TestRateLimitConfigWithBudget:
    """Test RateLimitConfig with budget configuration."""

    @pytest.mark.fast
    def test_config_with_default_budget(self):
        """Test RateLimitConfig has default budget."""
        config = RateLimitConfig()

        assert config.budget is not None
        assert isinstance(config.budget, RateLimitBudgetConfig)

    @pytest.mark.fast
    def test_config_with_nested_budget_object(self):
        """Test RateLimitConfig with nested RateLimitBudgetConfig object."""
        budget = RateLimitBudgetConfig(max_rotations=5, max_vpn_switches=1)
        config = RateLimitConfig(budget=budget)

        assert config.budget.max_rotations == 5
        assert config.budget.max_vpn_switches == 1

    @pytest.mark.fast
    def test_budget_roundtrip_yaml(self):
        """Test budget survives YAML roundtrip."""
        # Create original config
        original = RateLimitConfig(
            slots_per_second=0.75,
            budget={
                'max_rotations': 7,
                'max_backoff_time': 450.0,
                'max_vpn_switches': 2
            }
        )

        # Serialize to dict (simulating YAML dump)
        data = {
            'slots_per_second': original.slots_per_second,
            'burst_size': original.burst_size,
            'jitter_factor': original.jitter_factor,
            'max_backoff_seconds': original.max_backoff_seconds,
            'budget': {
                'max_rotations': original.budget.max_rotations,
                'max_backoff_time': original.budget.max_backoff_time,
                'max_vpn_switches': original.budget.max_vpn_switches
            }
        }

        # Deserialize (simulating YAML load + __post_init__)
        restored = RateLimitConfig(**data)

        assert restored.slots_per_second == 0.75
        assert restored.budget.max_rotations == 7
        assert restored.budget.max_backoff_time == 450.0
        assert restored.budget.max_vpn_switches == 2
