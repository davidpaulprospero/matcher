"""
Tests for cross-healer coordination events (US-64-011).

Verifies:
- HealerEvent enum has required event types
- subscribe_event() and publish_event() on HealingOrchestrator
- _notify_healers() uses event system
- Healers receive and handle events from other healers
- Event handlers in APIHealer, DownloadHealer, CheckpointHealer
"""

import pytest
from unittest.mock import Mock, patch

from src.agents.base import (
    Healer,
    HealerResult,
    HealerAction,
    HealerEvent,
    HealerEventData,
)
from src.agents.orchestrator import HealingOrchestrator
from src.agents.strategy import HealingStrategy
from src.agents.healers.api import APIHealer
from src.agents.healers.checkpoint import CheckpointHealer
from src.agents.healers.download import DownloadHealer


class TestHealerEventEnum:
    """Test HealerEvent enum has required event types."""

    @pytest.mark.fast
    def test_config_changed_event_exists(self):
        assert HealerEvent.CONFIG_CHANGED.value == "config_changed"

    @pytest.mark.fast
    def test_cache_cleared_event_exists(self):
        assert HealerEvent.CACHE_CLEARED.value == "cache_cleared"

    @pytest.mark.fast
    def test_rate_limited_event_exists(self):
        assert HealerEvent.RATE_LIMITED.value == "rate_limited"

    @pytest.mark.fast
    def test_provider_switched_event_exists(self):
        assert HealerEvent.PROVIDER_SWITCHED.value == "provider_switched"


class TestHealerEventData:
    """Test HealerEventData dataclass."""

    @pytest.mark.fast
    def test_create_event_data(self):
        data = HealerEventData(
            event=HealerEvent.CONFIG_CHANGED,
            source_healer="api-healer",
            details={"timeout": 60},
        )
        assert data.event == HealerEvent.CONFIG_CHANGED
        assert data.source_healer == "api-healer"
        assert data.details == {"timeout": 60}

    @pytest.mark.fast
    def test_event_data_defaults(self):
        data = HealerEventData(
            event=HealerEvent.CACHE_CLEARED,
            source_healer="disk-healer",
        )
        assert data.details == {}


class TestEventSubscription:
    """Test subscribe_event() and publish_event() on HealingOrchestrator."""

    @pytest.mark.fast
    def test_subscribe_event(self, mock_config, project_dir):
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        # Verify subscription registry is initialized
        assert HealerEvent.CONFIG_CHANGED in orchestrator._event_subscriptions

    @pytest.mark.fast
    def test_subscribe_adds_healer(self, mock_config, project_dir):
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        orchestrator.subscribe_event(HealerEvent.RATE_LIMITED, "test-healer")
        assert "test-healer" in orchestrator._event_subscriptions[HealerEvent.RATE_LIMITED]

    @pytest.mark.fast
    def test_subscribe_no_duplicates(self, mock_config, project_dir):
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        orchestrator.subscribe_event(HealerEvent.RATE_LIMITED, "test-healer")
        orchestrator.subscribe_event(HealerEvent.RATE_LIMITED, "test-healer")
        count = orchestrator._event_subscriptions[HealerEvent.RATE_LIMITED].count("test-healer")
        assert count == 1

    @pytest.mark.fast
    def test_publish_event_delivers_to_subscribers(self, mock_config, project_dir):
        """Test that publish_event delivers events to subscribed healers."""
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )

        # Create a mock healer and register it
        mock_healer = Mock(spec=Healer)
        mock_healer.name = "mock-healer"
        orchestrator._healer_instances["mock-healer"] = mock_healer

        # Subscribe and publish
        orchestrator.subscribe_event(HealerEvent.CONFIG_CHANGED, "mock-healer")
        orchestrator.publish_event(
            HealerEvent.CONFIG_CHANGED, "api-healer", timeout=60
        )

        # Verify handle_event was called
        mock_healer.handle_event.assert_called_once()
        event_data = mock_healer.handle_event.call_args[0][0]
        assert event_data.event == HealerEvent.CONFIG_CHANGED
        assert event_data.source_healer == "api-healer"
        assert event_data.details == {"timeout": 60}

    @pytest.mark.fast
    def test_publish_event_skips_source_healer(self, mock_config, project_dir):
        """Test that source healer does not receive its own event."""
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )

        mock_healer = Mock(spec=Healer)
        mock_healer.name = "self-healer"
        orchestrator._healer_instances["self-healer"] = mock_healer

        orchestrator.subscribe_event(HealerEvent.CONFIG_CHANGED, "self-healer")
        orchestrator.publish_event(HealerEvent.CONFIG_CHANGED, "self-healer")

        mock_healer.handle_event.assert_not_called()

    @pytest.mark.fast
    def test_publish_event_handles_handler_exception(self, mock_config, project_dir):
        """Test that a failing handler doesn't crash the event system."""
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )

        mock_healer = Mock(spec=Healer)
        mock_healer.name = "bad-healer"
        mock_healer.handle_event.side_effect = RuntimeError("handler crash")
        orchestrator._healer_instances["bad-healer"] = mock_healer

        orchestrator.subscribe_event(HealerEvent.CONFIG_CHANGED, "bad-healer")

        # Should not raise
        orchestrator.publish_event(HealerEvent.CONFIG_CHANGED, "api-healer")


class TestNotifyHealersUsesEvents:
    """Test that _notify_healers() uses the event system."""

    @pytest.mark.fast
    def test_config_changed_publishes_event(self, mock_config, project_dir):
        """Test _notify_healers publishes CONFIG_CHANGED for modified_config results."""
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )

        with patch.object(orchestrator, 'publish_event') as mock_publish:
            result = HealerResult.config_changed("Increased timeout")
            orchestrator._notify_healers("api-healer", result)

            mock_publish.assert_any_call(
                HealerEvent.CONFIG_CHANGED,
                "api-healer",
                result_message="Increased timeout",
            )

    @pytest.mark.fast
    def test_disk_healer_publishes_cache_cleared(self, mock_config, project_dir):
        """Test _notify_healers publishes CACHE_CLEARED when disk healer acts."""
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )

        with patch.object(orchestrator, 'publish_event') as mock_publish:
            result = HealerResult.fixed("Cleaned cache", action=HealerAction.RETRY)
            orchestrator._notify_healers("disk-healer", result)

            mock_publish.assert_called_with(
                HealerEvent.CACHE_CLEARED,
                "disk-healer",
                result_message="Cleaned cache",
            )

    @pytest.mark.fast
    def test_non_config_result_no_config_event(self, mock_config, project_dir):
        """Test _notify_healers doesn't publish CONFIG_CHANGED for non-config results."""
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )

        with patch.object(orchestrator, 'publish_event') as mock_publish:
            result = HealerResult.fixed("Retried", action=HealerAction.RETRY)
            orchestrator._notify_healers("api-healer", result)

            # Should not have published CONFIG_CHANGED
            config_calls = [
                c for c in mock_publish.call_args_list
                if c[0][0] == HealerEvent.CONFIG_CHANGED
            ]
            assert len(config_calls) == 0


class TestHealerEventHandlers:
    """Test event handlers in concrete healers."""

    @pytest.mark.fast
    def test_api_healer_resets_backoff_on_config_changed(self, mock_config, project_dir):
        """API healer resets backoff when CONFIG_CHANGED event received."""
        healer = APIHealer(mock_config, project_dir)
        healer.backoff_time = 60.0
        healer.retry_count = 5

        event_data = HealerEventData(
            event=HealerEvent.CONFIG_CHANGED,
            source_healer="download-healer",
        )
        healer.handle_event(event_data)

        assert healer.backoff_time == healer.INITIAL_BACKOFF
        assert healer.retry_count == 0

    @pytest.mark.fast
    def test_api_healer_increases_backoff_on_rate_limited(self, mock_config, project_dir):
        """API healer increases backoff when RATE_LIMITED event received."""
        healer = APIHealer(mock_config, project_dir)
        original_backoff = healer.backoff_time

        event_data = HealerEventData(
            event=HealerEvent.RATE_LIMITED,
            source_healer="download-healer",
        )
        healer.handle_event(event_data)

        assert healer.backoff_time > original_backoff

    @pytest.mark.fast
    def test_api_healer_resets_backoff_on_provider_switched(self, mock_config, project_dir):
        """API healer resets backoff when PROVIDER_SWITCHED event received."""
        healer = APIHealer(mock_config, project_dir)
        healer.backoff_time = 100.0
        healer.retry_count = 3

        event_data = HealerEventData(
            event=HealerEvent.PROVIDER_SWITCHED,
            source_healer="api-healer",
        )
        healer.handle_event(event_data)

        assert healer.backoff_time == healer.INITIAL_BACKOFF
        assert healer.retry_count == 0

    @pytest.mark.fast
    def test_download_healer_resets_backoff_on_config_changed(self, mock_config, project_dir):
        """Download healer resets backoff when CONFIG_CHANGED event received."""
        healer = DownloadHealer(mock_config, project_dir)
        healer.backoff_time = 120.0
        healer.retry_count = 4

        event_data = HealerEventData(
            event=HealerEvent.CONFIG_CHANGED,
            source_healer="api-healer",
        )
        healer.handle_event(event_data)

        assert healer.backoff_time == healer.INITIAL_BACKOFF
        assert healer.retry_count == 0

    @pytest.mark.fast
    def test_download_healer_increases_backoff_on_rate_limited(self, mock_config, project_dir):
        """Download healer increases backoff when RATE_LIMITED event received."""
        healer = DownloadHealer(mock_config, project_dir)
        original_backoff = healer.backoff_time

        event_data = HealerEventData(
            event=HealerEvent.RATE_LIMITED,
            source_healer="api-healer",
        )
        healer.handle_event(event_data)

        assert healer.backoff_time > original_backoff

    @pytest.mark.fast
    def test_checkpoint_healer_tracks_cache_cleared(self, mock_config, project_dir):
        """Checkpoint healer sets cache_was_cleaned on CACHE_CLEARED event."""
        healer = CheckpointHealer(mock_config, project_dir)
        assert healer.cache_was_cleaned is False

        event_data = HealerEventData(
            event=HealerEvent.CACHE_CLEARED,
            source_healer="disk-healer",
        )
        healer.handle_event(event_data)

        assert healer.cache_was_cleaned is True


class TestAutoSubscription:
    """Test that healers are auto-subscribed to relevant events during init."""

    @pytest.mark.fast
    def test_api_healer_subscribed_to_rate_limited(self, mock_config, project_dir):
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.conservative()
        )
        assert "api-healer" in orchestrator._event_subscriptions[HealerEvent.RATE_LIMITED]

    @pytest.mark.fast
    def test_api_healer_subscribed_to_config_changed(self, mock_config, project_dir):
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.conservative()
        )
        assert "api-healer" in orchestrator._event_subscriptions[HealerEvent.CONFIG_CHANGED]

    @pytest.mark.fast
    def test_checkpoint_healer_subscribed_to_cache_cleared(self, mock_config, project_dir):
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.conservative()
        )
        assert "checkpoint-healer" in orchestrator._event_subscriptions[HealerEvent.CACHE_CLEARED]

    @pytest.mark.fast
    def test_download_healer_subscribed_to_rate_limited(self, mock_config, project_dir):
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.conservative()
        )
        assert "download-healer" in orchestrator._event_subscriptions[HealerEvent.RATE_LIMITED]


class TestEndToEndEventDelivery:
    """Test full event flow: healer action -> _notify_healers -> event delivery."""

    @pytest.mark.fast
    def test_config_change_reaches_api_healer(self, mock_config, project_dir):
        """When _notify_healers is called with config change, API healer's backoff resets."""
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.conservative()
        )

        # Increase API healer backoff
        api_healer = orchestrator._healer_instances.get("api-healer")
        if api_healer:
            api_healer.backoff_time = 100.0
            api_healer.retry_count = 5

            # Simulate a config change from download healer
            result = HealerResult.config_changed("Socket timeout increased")
            orchestrator._notify_healers("download-healer", result)

            assert api_healer.backoff_time == api_healer.INITIAL_BACKOFF
            assert api_healer.retry_count == 0

    @pytest.mark.fast
    def test_cache_clear_reaches_checkpoint_healer(self, mock_config, project_dir):
        """When disk healer clears cache, checkpoint healer is notified."""
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.conservative()
        )

        checkpoint_healer = orchestrator._healer_instances.get("checkpoint-healer")
        if checkpoint_healer:
            assert checkpoint_healer.cache_was_cleaned is False

            result = HealerResult.fixed("Cleaned up cache", action=HealerAction.RETRY)
            orchestrator._notify_healers("disk-healer", result)

            assert checkpoint_healer.cache_was_cleaned is True
