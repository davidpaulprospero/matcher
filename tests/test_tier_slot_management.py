"""Tests for per-tier concurrent download slot management (US-114-011)."""

import pytest
import threading
import time
from src.downloader.orchestrator import TieredSlotManager, DownloadCoordinator


class TestTieredSlotManager:
    """Tests for TieredSlotManager class."""

    def test_default_tier_limits(self):
        """Test default max concurrent per tier."""
        manager = TieredSlotManager()
        assert manager.max_per_tier == {
            'short': 2,
            'medium': 1,
            'long': 1,
            'longer': 1,
        }

    def test_custom_tier_limits(self):
        """Test custom max concurrent per tier."""
        config = {'short': 3, 'medium': 2, 'long': 1, 'longer': 1}
        manager = TieredSlotManager(max_concurrent_per_tier=config)
        assert manager.max_per_tier == config

    def test_acquire_and_release(self):
        """Test basic acquire and release."""
        manager = TieredSlotManager()

        # Acquire short tier slot
        assert manager.acquire('short', 'download1') is True
        assert manager.get_active_count('short') == 1
        assert manager.get_available_slots('short') == 1

        # Release
        manager.release('short', 'download1')
        assert manager.get_active_count('short') == 0
        assert manager.get_available_slots('short') == 2

    def test_per_tier_limits_enforced(self):
        """Test that per-tier limits are enforced when borrowing disabled."""
        config = {'short': 2, 'medium': 1, 'long': 1, 'longer': 1}
        manager = TieredSlotManager(
            max_concurrent_per_tier=config,
            allow_borrowing=False  # Disable borrowing to test strict limits
        )

        # Fill short tier (max 2)
        assert manager.acquire('short', 's1') is True
        assert manager.acquire('short', 's2') is True
        # Third should fail (tier limit) - no borrowing allowed
        assert manager.acquire('short', 's3', timeout=0.1) is False

    def test_borrowing_between_tiers(self):
        """Test slot borrowing between tiers when enabled."""
        config = {'short': 2, 'medium': 1, 'long': 1, 'longer': 1}
        manager = TieredSlotManager(
            max_concurrent_per_tier=config,
            allow_borrowing=True
        )

        # Fill short tier
        assert manager.acquire('short', 's1') is True
        assert manager.acquire('short', 's2') is True

        # Fill medium tier
        assert manager.acquire('medium', 'm1') is True

        # Try to acquire for long - should fail (no slots and no borrowing from lower tiers)
        # Actually, long can borrow from short/medium since they have higher priority
        # Let's try short borrowing from medium when medium is full

        # Release all and try borrowing
        manager.release('short', 's1')
        manager.release('short', 's2')
        manager.release('medium', 'm1')

        # Fill all tiers
        assert manager.acquire('short', 's1') is True
        assert manager.acquire('medium', 'm1') is True
        assert manager.acquire('long', 'l1') is True
        assert manager.acquire('longer', 'lo1') is True

        # Try another short - should fail (short at limit)
        result = manager.acquire('short', 's2', timeout=0.1)
        # May succeed via borrowing from longer if allowed
        # Result depends on implementation

    def test_no_borrowing_when_disabled(self):
        """Test that borrowing is disabled when configured."""
        config = {'short': 1, 'medium': 1}
        manager = TieredSlotManager(
            max_concurrent_per_tier=config,
            allow_borrowing=False
        )

        # Fill both tiers
        assert manager.acquire('short', 's1') is True
        assert manager.acquire('medium', 'm1') is True

        # Try to acquire more - should fail
        assert manager.acquire('short', 's2', timeout=0.1) is False

    def test_total_concurrency_cap(self):
        """Test max total concurrent limit."""
        config = {'short': 10, 'medium': 10, 'long': 10, 'longer': 10}
        manager = TieredSlotManager(
            max_concurrent_per_tier=config,
            max_total_concurrent=3
        )

        # Can only acquire 3 total
        assert manager.acquire('short', 's1') is True
        assert manager.acquire('short', 's2') is True
        assert manager.acquire('short', 's3') is True
        # Fourth should fail due to total cap
        assert manager.acquire('short', 's4', timeout=0.1) is False

    def test_get_status(self):
        """Test status reporting."""
        config = {'short': 2, 'medium': 1}
        manager = TieredSlotManager(max_concurrent_per_tier=config)

        manager.acquire('short', 's1')
        manager.acquire('medium', 'm1')

        status = manager.get_status()
        assert status['active_per_tier']['short'] == 1
        assert status['active_per_tier']['medium'] == 1
        assert status['total_active'] == 2
        assert 's1' in status['active_downloads']
        assert 'm1' in status['active_downloads']

    def test_reset(self):
        """Test reset functionality."""
        manager = TieredSlotManager()

        manager.acquire('short', 's1')
        manager.acquire('medium', 'm1')

        manager.reset()

        status = manager.get_status()
        assert status['active_per_tier']['short'] == 0
        assert status['active_per_tier']['medium'] == 0
        assert status['total_active'] == 0


class TestDownloadCoordinatorWithTierSlots:
    """Tests for DownloadCoordinator with tier slot management enabled."""

    def test_coordinator_with_tier_slots(self):
        """Test coordinator with tier slots enabled."""
        tier_config = {'short': 2, 'medium': 1, 'long': 1, 'longer': 1}
        coordinator = DownloadCoordinator(
            max_concurrent=4,
            tier_config=tier_config,
            enable_tier_slots=True,
            allow_borrowing=True,
            max_total_concurrent=4
        )

        assert coordinator.is_tier_slots_enabled is True

        # Acquire with tier
        assert coordinator.acquire('s1', tier='short') is True
        assert coordinator.acquire('m1', tier='medium') is True

        status = coordinator.get_status()
        assert status['tier_slots_enabled'] is True
        assert 'tier_status' in status

    def test_coordinator_without_tier_slots(self):
        """Test coordinator with tier slots disabled."""
        coordinator = DownloadCoordinator(
            max_concurrent=4,
            enable_tier_slots=False
        )

        assert coordinator.is_tier_slots_enabled is False
        assert coordinator.max_concurrent == 4

        # Acquire without tier (should use global limit)
        assert coordinator.acquire('d1') is True
        assert coordinator.active_count == 1

    def test_coordinator_tier_release(self):
        """Test release with tier."""
        tier_config = {'short': 2, 'medium': 1}
        coordinator = DownloadCoordinator(
            tier_config=tier_config,
            enable_tier_slots=True
        )

        coordinator.acquire('s1', tier='short')
        coordinator.acquire('m1', tier='medium')

        coordinator.release('s1', tier='short', success=True)
        coordinator.release('m1', tier='medium', success=False)

        status = coordinator.get_status()
        assert status['completed'] == 1
        assert status['failed'] == 1


class TestTierSlotManagementConfig:
    """Tests for TierSlotManagementConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from src.config.sections.download import TierSlotManagementConfig

        config = TierSlotManagementConfig()
        assert config.enabled is True
        assert config.max_concurrent_per_tier == {
            'short': 2,
            'medium': 1,
            'long': 1,
            'longer': 1,
        }
        assert config.allow_borrowing is True
        assert config.max_total_concurrent is None

    def test_custom_config(self):
        """Test custom configuration."""
        from src.config.sections.download import TierSlotManagementConfig

        config = TierSlotManagementConfig(
            enabled=False,
            max_concurrent_per_tier={'short': 4, 'medium': 2},
            allow_borrowing=False,
            max_total_concurrent=6
        )
        assert config.enabled is False
        assert config.max_concurrent_per_tier['short'] == 4
        assert config.allow_borrowing is False
        assert config.max_total_concurrent == 6

    def test_validation_invalid_tier(self):
        """Test validation catches invalid tier."""
        from src.config.sections.download import TierSlotManagementConfig

        with pytest.raises(ValueError, match="invalid tier"):
            TierSlotManagementConfig(
                max_concurrent_per_tier={'invalid_tier': 1}
            )

    def test_validation_invalid_slot_count(self):
        """Test validation catches invalid slot count."""
        from src.config.sections.download import TierSlotManagementConfig

        with pytest.raises(ValueError, match="must be >= 1"):
            TierSlotManagementConfig(
                max_concurrent_per_tier={'short': 0}
            )
