"""Tests for circuit breaker cascade integration (US-61-003).

Verifies that caption and download circuit breakers coordinate their state
when YouTube is rate-limiting, so both systems pause together.
"""

import pytest
import time
from unittest.mock import MagicMock, patch

from src.caption.circuit_breaker import (
    CaptionCircuitBreaker,
    CaptionCircuitBreakerConfig,
    CaptionCircuitBreakerState,
)
from src.downloader.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
)


class TestCaptionCircuitBreakerCascadeConfig:
    """Tests for circuit_breaker_cascade config option on caption CB."""

    @pytest.mark.fast
    def test_default_cascade_enabled(self):
        """circuit_breaker_cascade should default to True."""
        config = CaptionCircuitBreakerConfig()
        assert config.circuit_breaker_cascade is True

    @pytest.mark.fast
    def test_cascade_can_be_disabled(self):
        """circuit_breaker_cascade can be set to False."""
        config = CaptionCircuitBreakerConfig(circuit_breaker_cascade=False)
        assert config.circuit_breaker_cascade is False


class TestDownloadCircuitBreakerCascadeConfig:
    """Tests for circuit_breaker_cascade config option on download CB."""

    @pytest.mark.fast
    def test_default_cascade_enabled(self):
        """circuit_breaker_cascade should default to True."""
        config = CircuitBreakerConfig()
        assert config.circuit_breaker_cascade is True

    @pytest.mark.fast
    def test_cascade_can_be_disabled(self):
        """circuit_breaker_cascade can be set to False."""
        config = CircuitBreakerConfig(circuit_breaker_cascade=False)
        assert config.circuit_breaker_cascade is False


class TestCaptionCBSetDownloadCB:
    """Tests for set_download_circuit_breaker() method."""

    @pytest.mark.fast
    def test_links_download_cb_instance(self):
        """set_download_circuit_breaker() should store download CB reference."""
        caption_cb = CaptionCircuitBreaker()
        download_cb = CircuitBreaker()

        caption_cb.set_download_circuit_breaker(download_cb)

        assert caption_cb._download_circuit_breaker is download_cb

    @pytest.mark.fast
    def test_without_link_is_none(self):
        """Without set_download_circuit_breaker(), reference should be None."""
        caption_cb = CaptionCircuitBreaker()
        assert caption_cb._download_circuit_breaker is None


class TestDownloadCBSetCaptionCB:
    """Tests for set_caption_circuit_breaker() method."""

    @pytest.mark.fast
    def test_links_caption_cb_instance(self):
        """set_caption_circuit_breaker() should store caption CB reference."""
        caption_cb = CaptionCircuitBreaker()
        download_cb = CircuitBreaker()

        download_cb._caption_circuit_breaker = caption_cb

        assert download_cb._caption_circuit_breaker is caption_cb

    @pytest.mark.fast
    def test_without_link_is_none(self):
        """Without set_caption_circuit_breaker(), reference should be None."""
        download_cb = CircuitBreaker()
        assert download_cb._caption_circuit_breaker is None


class TestCaptionCBCascadeFailure:
    """Tests for caption CB cascading failures to download CB."""

    @pytest.mark.fast
    def test_failure_increments_download_cb_counter(self):
        """Caption CB failure should increment download CB failure counter."""
        caption_cb = CaptionCircuitBreaker(CaptionCircuitBreakerConfig(threshold=5))
        download_cb = CircuitBreaker(CircuitBreakerConfig(consecutive_failures_threshold=5))
        caption_cb.set_download_circuit_breaker(download_cb)

        assert download_cb.state.consecutive_failures == 0

        caption_cb.record_failure()

        # Both should have 1 failure
        assert caption_cb.state.consecutive_failures == 1
        assert download_cb.state.consecutive_failures == 1

    @pytest.mark.fast
    def test_cascade_disabled_does_not_propagate(self):
        """When cascade disabled, failures don't propagate."""
        caption_cb = CaptionCircuitBreaker(CaptionCircuitBreakerConfig(
            threshold=5,
            circuit_breaker_cascade=False
        ))
        download_cb = CircuitBreaker(CircuitBreakerConfig(consecutive_failures_threshold=5))
        caption_cb.set_download_circuit_breaker(download_cb)

        caption_cb.record_failure()

        assert caption_cb.state.consecutive_failures == 1
        assert download_cb.state.consecutive_failures == 0  # Not incremented

    @pytest.mark.fast
    def test_no_cascade_without_link(self):
        """Without linked download CB, failure doesn't crash."""
        caption_cb = CaptionCircuitBreaker(CaptionCircuitBreakerConfig(threshold=5))

        # Should not raise
        caption_cb.record_failure()
        assert caption_cb.state.consecutive_failures == 1


class TestDownloadCBCascadeFailure:
    """Tests for download CB cascading failures to caption CB."""

    @pytest.mark.fast
    def test_failure_increments_caption_cb_counter(self):
        """Download CB failure should increment caption CB failure counter."""
        caption_cb = CaptionCircuitBreaker(CaptionCircuitBreakerConfig(threshold=5))
        download_cb = CircuitBreaker(CircuitBreakerConfig(consecutive_failures_threshold=5))
        download_cb._caption_circuit_breaker = caption_cb

        assert caption_cb.state.consecutive_failures == 0

        download_cb.record_failure()

        # Both should have 1 failure
        assert download_cb.state.consecutive_failures == 1
        assert caption_cb.state.consecutive_failures == 1

    @pytest.mark.fast
    def test_cascade_disabled_does_not_propagate(self):
        """When cascade disabled, failures don't propagate."""
        caption_cb = CaptionCircuitBreaker(CaptionCircuitBreakerConfig(threshold=5))
        download_cb = CircuitBreaker(CircuitBreakerConfig(
            consecutive_failures_threshold=5,
            circuit_breaker_cascade=False
        ))
        download_cb._caption_circuit_breaker = caption_cb

        download_cb.record_failure()

        assert download_cb.state.consecutive_failures == 1
        assert caption_cb.state.consecutive_failures == 0  # Not incremented


class TestCaptionCBCascadeTrip:
    """Tests for caption CB cascading trip state to download CB."""

    @pytest.mark.fast
    def test_trip_propagates_to_download_cb(self):
        """When caption CB trips, download CB should also trip."""
        caption_cb = CaptionCircuitBreaker(CaptionCircuitBreakerConfig(threshold=2))
        download_cb = CircuitBreaker(CircuitBreakerConfig(consecutive_failures_threshold=5))
        caption_cb.set_download_circuit_breaker(download_cb)

        # Record enough failures to trip caption CB
        caption_cb.record_failure()  # 1
        tripped = caption_cb.record_failure()  # 2 - trips!

        assert tripped is True
        assert caption_cb.state.is_open is True
        assert download_cb.state.is_open is True  # Cascaded!
        assert download_cb.state.total_trips == 1

    @pytest.mark.fast
    def test_trip_cascade_disabled_does_not_propagate(self):
        """When cascade disabled, trip doesn't propagate."""
        caption_cb = CaptionCircuitBreaker(CaptionCircuitBreakerConfig(
            threshold=2,
            circuit_breaker_cascade=False
        ))
        download_cb = CircuitBreaker(CircuitBreakerConfig(consecutive_failures_threshold=5))
        caption_cb.set_download_circuit_breaker(download_cb)

        caption_cb.record_failure()
        caption_cb.record_failure()  # Trips

        assert caption_cb.state.is_open is True
        assert download_cb.state.is_open is False  # Not cascaded

    @pytest.mark.fast
    def test_trip_shares_opened_at_timestamp(self):
        """Cascaded trip should use the same opened_at timestamp."""
        caption_cb = CaptionCircuitBreaker(CaptionCircuitBreakerConfig(threshold=2))
        download_cb = CircuitBreaker(CircuitBreakerConfig(consecutive_failures_threshold=5))
        caption_cb.set_download_circuit_breaker(download_cb)

        caption_cb.record_failure()
        caption_cb.record_failure()

        assert caption_cb.state.opened_at == download_cb.state.opened_at


class TestDownloadCBCascadeTrip:
    """Tests for download CB cascading trip state to caption CB."""

    @pytest.mark.fast
    def test_trip_propagates_to_caption_cb(self):
        """When download CB trips, caption CB should also trip."""
        caption_cb = CaptionCircuitBreaker(CaptionCircuitBreakerConfig(threshold=10))
        download_cb = CircuitBreaker(CircuitBreakerConfig(consecutive_failures_threshold=2))
        download_cb._caption_circuit_breaker = caption_cb

        # Record enough failures to trip download CB
        download_cb.record_failure()  # 1
        tripped = download_cb.record_failure()  # 2 - trips!

        assert tripped is True
        assert download_cb.state.is_open is True
        assert caption_cb.state.is_open is True  # Cascaded!
        assert caption_cb.state.total_trips == 1

    @pytest.mark.fast
    def test_trip_cascade_disabled_does_not_propagate(self):
        """When cascade disabled, trip doesn't propagate."""
        caption_cb = CaptionCircuitBreaker(CaptionCircuitBreakerConfig(threshold=10))
        download_cb = CircuitBreaker(CircuitBreakerConfig(
            consecutive_failures_threshold=2,
            circuit_breaker_cascade=False
        ))
        download_cb._caption_circuit_breaker = caption_cb

        download_cb.record_failure()
        download_cb.record_failure()  # Trips

        assert download_cb.state.is_open is True
        assert caption_cb.state.is_open is False  # Not cascaded


class TestBidirectionalCascade:
    """Tests for bidirectional cascade with both CBs linked."""

    @pytest.mark.fast
    def test_both_cbs_linked_bidirectionally(self):
        """Both CBs can be linked to each other for full cascade."""
        caption_cb = CaptionCircuitBreaker(CaptionCircuitBreakerConfig(threshold=5))
        download_cb = CircuitBreaker(CircuitBreakerConfig(consecutive_failures_threshold=5))

        # Link both ways
        caption_cb.set_download_circuit_breaker(download_cb)
        download_cb._caption_circuit_breaker = caption_cb

        # Caption failure affects both
        caption_cb.record_failure()
        assert caption_cb.state.consecutive_failures == 1
        assert download_cb.state.consecutive_failures == 1

        # Download failure affects both (but caption already has 1)
        download_cb.record_failure()
        assert download_cb.state.consecutive_failures == 2
        assert caption_cb.state.consecutive_failures == 2

    @pytest.mark.fast
    def test_caption_trip_opens_both_when_bidirectional(self):
        """Caption CB trip should open both when linked bidirectionally."""
        caption_cb = CaptionCircuitBreaker(CaptionCircuitBreakerConfig(threshold=2))
        download_cb = CircuitBreaker(CircuitBreakerConfig(consecutive_failures_threshold=5))

        caption_cb.set_download_circuit_breaker(download_cb)
        download_cb._caption_circuit_breaker = caption_cb

        caption_cb.record_failure()
        caption_cb.record_failure()  # Trips caption CB

        assert caption_cb.state.is_open is True
        assert download_cb.state.is_open is True

    @pytest.mark.fast
    def test_download_trip_opens_both_when_bidirectional(self):
        """Download CB trip should open both when linked bidirectionally."""
        caption_cb = CaptionCircuitBreaker(CaptionCircuitBreakerConfig(threshold=10))
        download_cb = CircuitBreaker(CircuitBreakerConfig(consecutive_failures_threshold=2))

        caption_cb.set_download_circuit_breaker(download_cb)
        download_cb._caption_circuit_breaker = caption_cb

        download_cb.record_failure()
        download_cb.record_failure()  # Trips download CB

        assert download_cb.state.is_open is True
        assert caption_cb.state.is_open is True


class TestCascadeWithDisabledCB:
    """Tests for cascade when one CB is disabled."""

    @pytest.mark.fast
    def test_caption_cascade_to_disabled_download_cb_skipped(self):
        """Cascade to disabled download CB should be skipped."""
        caption_cb = CaptionCircuitBreaker(CaptionCircuitBreakerConfig(threshold=2))
        download_cb = CircuitBreaker(CircuitBreakerConfig(enabled=False))
        caption_cb.set_download_circuit_breaker(download_cb)

        caption_cb.record_failure()
        caption_cb.record_failure()  # Trips

        assert caption_cb.state.is_open is True
        assert download_cb.state.is_open is False  # Disabled CB not affected

    @pytest.mark.fast
    def test_download_cascade_to_disabled_caption_cb_skipped(self):
        """Cascade to disabled caption CB should be skipped."""
        caption_cb = CaptionCircuitBreaker(CaptionCircuitBreakerConfig(enabled=False))
        download_cb = CircuitBreaker(CircuitBreakerConfig(consecutive_failures_threshold=2))
        download_cb._caption_circuit_breaker = caption_cb

        download_cb.record_failure()
        download_cb.record_failure()  # Trips

        assert download_cb.state.is_open is True
        assert caption_cb.state.is_open is False  # Disabled CB not affected


class TestConfigSectionsCascade:
    """Tests for config sections including circuit_breaker_cascade."""

    @pytest.mark.fast
    def test_caption_config_section_has_cascade(self):
        """CaptionCircuitBreakerConfig in config sections should have cascade."""
        from src.config.sections.download import CaptionCircuitBreakerConfig as ConfigCaptionCBConfig
        config = ConfigCaptionCBConfig()
        assert hasattr(config, 'circuit_breaker_cascade')
        assert config.circuit_breaker_cascade is True

    @pytest.mark.fast
    def test_download_config_section_has_cascade(self):
        """CircuitBreakerConfig in config sections should have cascade."""
        from src.config.sections.download import CircuitBreakerConfig as ConfigDownloadCBConfig
        config = ConfigDownloadCBConfig()
        assert hasattr(config, 'circuit_breaker_cascade')
        assert config.circuit_breaker_cascade is True


class TestCascadeDoesNotDoubleTrip:
    """Tests to ensure cascade doesn't cause infinite loops or double trips."""

    @pytest.mark.fast
    def test_already_open_cb_not_tripped_again(self):
        """If target CB is already open, cascade trip is a no-op."""
        caption_cb = CaptionCircuitBreaker(CaptionCircuitBreakerConfig(threshold=2))
        download_cb = CircuitBreaker(CircuitBreakerConfig(consecutive_failures_threshold=2))
        caption_cb.set_download_circuit_breaker(download_cb)

        # Manually open download CB
        download_cb.state.is_open = True
        download_cb.state.opened_at = time.time()
        download_cb.state.total_trips = 1
        original_trips = download_cb.state.total_trips

        # Now trip caption CB - should NOT increment download CB trips
        caption_cb.record_failure()
        caption_cb.record_failure()  # Trips

        assert caption_cb.state.total_trips == 1
        assert download_cb.state.total_trips == original_trips  # Not incremented
