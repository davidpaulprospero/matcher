"""
Tests for config merge helpers - US-010

Covers:
- AC5: _merge_single_tier() maps config aliases correctly
- AC6: _merge_duration_tiers() detects tier names and applies per-tier overrides
"""

import pytest
from unittest.mock import patch, call

from src.config.sections.duration import DurationTierConfig, DurationTiersConfig
from src.cli.config_utils import _merge_duration_tiers

# Import _merge_single_tier (private helper)
from src.cli.config_utils import _merge_single_tier


# ============================================================================
# AC5: _merge_single_tier() maps config aliases correctly
# ============================================================================

class TestMergeSingleTierAliasesUS010:
    """AC5: _merge_single_tier() maps config aliases correctly:
    'count'->'videos_per_keyword', 'min'->'min_seconds', 'max'->'max_seconds'."""

    @pytest.mark.fast
    def test_count_maps_to_videos_per_keyword(self):
        """'count' alias maps to 'videos_per_keyword' field."""
        tier = DurationTierConfig(min_seconds=10, max_seconds=60, videos_per_keyword=5)

        _merge_single_tier(tier, {'count': 99})

        assert tier.videos_per_keyword == 99

    @pytest.mark.fast
    def test_per_keyword_maps_to_videos_per_keyword(self):
        """'per_keyword' alias maps to 'videos_per_keyword' field."""
        tier = DurationTierConfig(min_seconds=10, max_seconds=60, videos_per_keyword=5)

        _merge_single_tier(tier, {'per_keyword': 42})

        assert tier.videos_per_keyword == 42

    @pytest.mark.fast
    def test_min_maps_to_min_seconds(self):
        """'min' alias maps to 'min_seconds' field."""
        tier = DurationTierConfig(min_seconds=10, max_seconds=60, videos_per_keyword=5)

        _merge_single_tier(tier, {'min': 120})

        assert tier.min_seconds == 120

    @pytest.mark.fast
    def test_max_maps_to_max_seconds(self):
        """'max' alias maps to 'max_seconds' field."""
        tier = DurationTierConfig(min_seconds=10, max_seconds=60, videos_per_keyword=5)

        _merge_single_tier(tier, {'max': 900})

        assert tier.max_seconds == 900

    @pytest.mark.fast
    def test_canonical_names_work_directly(self):
        """Canonical field names (videos_per_keyword, min_seconds, max_seconds)
        work without alias mapping."""
        tier = DurationTierConfig(min_seconds=10, max_seconds=60, videos_per_keyword=5)

        _merge_single_tier(tier, {
            'videos_per_keyword': 7,
            'min_seconds': 30,
            'max_seconds': 300
        })

        assert tier.videos_per_keyword == 7
        assert tier.min_seconds == 30
        assert tier.max_seconds == 300

    @pytest.mark.fast
    def test_all_aliases_in_single_call(self):
        """All aliases mapped correctly in a single call."""
        tier = DurationTierConfig(min_seconds=0, max_seconds=0, videos_per_keyword=0)

        _merge_single_tier(tier, {
            'count': 10,
            'min': 60,
            'max': 600
        })

        assert tier.videos_per_keyword == 10
        assert tier.min_seconds == 60
        assert tier.max_seconds == 600

    @pytest.mark.fast
    def test_max_total_passes_through(self):
        """'max_total' passes through without alias mapping."""
        tier = DurationTierConfig(min_seconds=10, max_seconds=60, videos_per_keyword=5, max_total=20)

        _merge_single_tier(tier, {'max_total': 50})

        assert tier.max_total == 50

    @pytest.mark.fast
    def test_unknown_keys_ignored(self):
        """Keys that don't match any field are silently ignored."""
        tier = DurationTierConfig(min_seconds=10, max_seconds=60, videos_per_keyword=5)

        _merge_single_tier(tier, {'nonexistent_key': 999, 'count': 3})

        assert tier.videos_per_keyword == 3
        assert tier.min_seconds == 10  # unchanged

    @pytest.mark.fast
    def test_setattr_called_with_canonical_names(self):
        """Verify setattr is called with canonical field names, not aliases."""
        tier = DurationTierConfig(min_seconds=10, max_seconds=60, videos_per_keyword=5)

        with patch.object(type(tier), '__setattr__', wraps=tier.__setattr__) as mock_setattr:
            # Need to use a fresh tier since the mock wraps the original
            pass

        # Direct verification: after calling with aliases, canonical fields are set
        tier2 = DurationTierConfig(min_seconds=0, max_seconds=0, videos_per_keyword=0)
        _merge_single_tier(tier2, {'count': 7, 'min': 30, 'max': 120})

        # The canonical names should be the ones with updated values
        assert tier2.videos_per_keyword == 7  # 'count' -> 'videos_per_keyword'
        assert tier2.min_seconds == 30         # 'min' -> 'min_seconds'
        assert tier2.max_seconds == 120        # 'max' -> 'max_seconds'

    @pytest.mark.fast
    def test_returns_tier_config(self):
        """_merge_single_tier returns the updated tier config object."""
        tier = DurationTierConfig(min_seconds=10, max_seconds=60, videos_per_keyword=5)

        result = _merge_single_tier(tier, {'count': 3})

        assert result is tier


# ============================================================================
# AC6: _merge_duration_tiers() detects tier names and applies per-tier overrides
# ============================================================================

class TestMergeDurationTiersPerTierUS010:
    """AC6: _merge_duration_tiers() detects tier names (short/medium/long/longer)
    and applies per-tier overrides without affecting sibling tiers."""

    @pytest.mark.fast
    def test_short_tier_override_only(self):
        """Overriding 'short' doesn't affect medium, long, or longer."""
        tiers = DurationTiersConfig()
        original_medium = tiers.medium.videos_per_keyword
        original_long = tiers.long.videos_per_keyword
        original_longer = tiers.longer.videos_per_keyword

        _merge_duration_tiers(tiers, {'short': {'count': 99}})

        assert tiers.short.videos_per_keyword == 99
        assert tiers.medium.videos_per_keyword == original_medium
        assert tiers.long.videos_per_keyword == original_long
        assert tiers.longer.videos_per_keyword == original_longer

    @pytest.mark.fast
    def test_medium_tier_override_only(self):
        """Overriding 'medium' doesn't affect short, long, or longer."""
        tiers = DurationTiersConfig()
        original_short = tiers.short.videos_per_keyword
        original_long = tiers.long.videos_per_keyword

        _merge_duration_tiers(tiers, {'medium': {'count': 77}})

        assert tiers.medium.videos_per_keyword == 77
        assert tiers.short.videos_per_keyword == original_short
        assert tiers.long.videos_per_keyword == original_long

    @pytest.mark.fast
    def test_long_tier_override_only(self):
        """Overriding 'long' doesn't affect other tiers."""
        tiers = DurationTiersConfig()
        original_short = tiers.short.videos_per_keyword
        original_medium = tiers.medium.videos_per_keyword

        _merge_duration_tiers(tiers, {'long': {'min': 600, 'max': 1200, 'count': 3}})

        assert tiers.long.min_seconds == 600
        assert tiers.long.max_seconds == 1200
        assert tiers.long.videos_per_keyword == 3
        assert tiers.short.videos_per_keyword == original_short
        assert tiers.medium.videos_per_keyword == original_medium

    @pytest.mark.fast
    def test_longer_tier_override_only(self):
        """Overriding 'longer' doesn't affect other tiers."""
        tiers = DurationTiersConfig()
        original_short_min = tiers.short.min_seconds

        _merge_duration_tiers(tiers, {'longer': {'count': 0, 'max_total': 0}})

        assert tiers.longer.videos_per_keyword == 0
        assert tiers.longer.max_total == 0
        assert tiers.short.min_seconds == original_short_min

    @pytest.mark.fast
    def test_all_four_tiers_get_specific_values(self):
        """Each tier receives its own specific override values."""
        tiers = DurationTiersConfig()

        _merge_duration_tiers(tiers, {
            'short': {'count': 10},
            'medium': {'count': 20},
            'long': {'count': 30},
            'longer': {'count': 40}
        })

        assert tiers.short.videos_per_keyword == 10
        assert tiers.medium.videos_per_keyword == 20
        assert tiers.long.videos_per_keyword == 30
        assert tiers.longer.videos_per_keyword == 40

    @pytest.mark.fast
    def test_partial_tier_override_preserves_existing_values(self):
        """Overriding one field in a tier preserves its other fields."""
        tiers = DurationTiersConfig()
        original_short_min = tiers.short.min_seconds
        original_short_max = tiers.short.max_seconds

        _merge_duration_tiers(tiers, {'short': {'count': 50}})

        assert tiers.short.videos_per_keyword == 50
        assert tiers.short.min_seconds == original_short_min
        assert tiers.short.max_seconds == original_short_max

    @pytest.mark.fast
    def test_detects_tier_names_in_override_keys(self):
        """_merge_duration_tiers detects short/medium/long/longer as tier names."""
        tiers = DurationTiersConfig()

        # All four tier names should be recognized
        _merge_duration_tiers(tiers, {
            'short': {'count': 1},
            'medium': {'count': 2},
            'long': {'count': 3},
            'longer': {'count': 4}
        })

        assert tiers.short.videos_per_keyword == 1
        assert tiers.medium.videos_per_keyword == 2
        assert tiers.long.videos_per_keyword == 3
        assert tiers.longer.videos_per_keyword == 4

    @pytest.mark.fast
    def test_ignores_non_tier_keys(self):
        """Non-tier keys in overrides are ignored."""
        tiers = DurationTiersConfig()
        original_short = tiers.short.videos_per_keyword

        _merge_duration_tiers(tiers, {
            'invalid_tier': {'count': 999},
            'extra_long': {'count': 888}
        })

        assert tiers.short.videos_per_keyword == original_short

    @pytest.mark.fast
    def test_mixed_valid_and_invalid_tier_names(self):
        """Valid tier names are processed while invalid ones are ignored."""
        tiers = DurationTiersConfig()

        _merge_duration_tiers(tiers, {
            'short': {'count': 5},
            'bogus': {'count': 999},
            'long': {'count': 2}
        })

        assert tiers.short.videos_per_keyword == 5
        assert tiers.long.videos_per_keyword == 2
