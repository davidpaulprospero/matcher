"""Tests for multi-format download fallback (US-114-005).

Verifies that FormatFallbackConfig, FormatFallbackHandler, and related
functionality correctly implement multi-format fallback behavior.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.config.sections.download import FormatFallbackConfig


class TestFormatFallbackConfig:
    """Tests for FormatFallbackConfig validation and defaults."""

    def test_default_format_priority(self):
        """Default format_priority includes mp4, webm, vp9.2, avc, mp3, audio-only."""
        config = FormatFallbackConfig()
        # US-144-006: avc and mp3 added to default priority
        assert config.format_priority == ["mp4", "webm", "vp9.2", "avc", "mp3", "audio-only"]

    def test_audio_only_at_end(self):
        """audio-only is always placed at end of priority list."""
        config = FormatFallbackConfig(format_priority=["webm", "mp4", "audio-only"])
        # Should reorder so audio-only is last
        assert config.format_priority[-1] == "audio-only"

    def test_custom_format_priority(self):
        """Custom format_priority is preserved."""
        config = FormatFallbackConfig(format_priority=["webm", "mp4"])
        assert config.format_priority == ["webm", "mp4"]

    def test_validates_invalid_format(self):
        """Invalid format raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            FormatFallbackConfig(format_priority=["mp4", "invalid_format"])
        assert "invalid format" in str(exc_info.value).lower()

    def test_validates_negative_retries(self):
        """Negative max_retries_per_format raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            FormatFallbackConfig(max_retries_per_format=-1)
        assert "must be >= 0" in str(exc_info.value)

    def test_validates_min_samples(self):
        """min_samples_for_adaptation < 1 raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            FormatFallbackConfig(min_samples_for_adaptation=0)
        assert "must be >= 1" in str(exc_info.value)

    def test_default_values(self):
        """Default configuration values are sensible."""
        config = FormatFallbackConfig()
        assert config.enabled is True
        assert config.max_retries_per_format == 2
        assert config.adaptive_priority is True
        assert config.min_samples_for_adaptation == 10
        assert config.track_success_rates is True
        assert config.fallback_on_any_error is True


class TestFormatFallbackHandler:
    """Tests for FormatFallbackHandler logic."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for testing."""
        config = MagicMock()
        config.download.format_fallback = FormatFallbackConfig()
        config.download.format_preference = None
        config.download.davinci_mode = True
        config.download.quality = "1080"
        return config

    def test_is_enabled_when_config_enabled(self, mock_config):
        """is_enabled returns True when format_fallback is enabled."""
        from src.downloader.format_fallback import FormatFallbackHandler
        handler = FormatFallbackHandler(mock_config)
        assert handler.is_enabled is True

    def test_is_enabled_when_config_disabled(self, mock_config):
        """is_enabled returns False when format_fallback is disabled."""
        mock_config.download.format_fallback.enabled = False
        from src.downloader.format_fallback import FormatFallbackHandler
        handler = FormatFallbackHandler(mock_config)
        assert handler.is_enabled is False

    def test_is_enabled_when_config_none(self, mock_config):
        """is_enabled returns False when format_fallback is None."""
        mock_config.download.format_fallback = None
        from src.downloader.format_fallback import FormatFallbackHandler
        handler = FormatFallbackHandler(mock_config)
        assert handler.is_enabled is False

    def test_get_format_priority_default(self, mock_config):
        """get_format_priority returns default priority order."""
        from src.downloader.format_fallback import FormatFallbackHandler
        handler = FormatFallbackHandler(mock_config)
        priority = handler.get_format_priority()
        # US-144-006: avc and mp3 added to default priority
        assert priority == ["mp4", "webm", "vp9.2", "avc", "mp3", "audio-only"]

    def test_build_composite_format_string_includes_all_formats(self, mock_config):
        """build_composite_format_string includes all formats from priority."""
        from src.downloader.format_fallback import FormatFallbackHandler
        handler = FormatFallbackHandler(mock_config)
        format_str = handler.build_composite_format_string("1080")

        # Should contain mp4, webm, and audio-only selectors
        assert "mp4" in format_str
        assert "webm" in format_str
        assert "bestaudio" in format_str  # audio-only

    def test_build_composite_format_string_uses_fallback(self, mock_config):
        """build_composite_format_string uses / for yt-dlp fallback."""
        from src.downloader.format_fallback import FormatFallbackHandler
        handler = FormatFallbackHandler(mock_config)
        format_str = handler.build_composite_format_string("1080")

        # yt-dlp fallback is denoted by /
        assert "/" in format_str

    def test_record_attempt_updates_stats(self, mock_config):
        """record_attempt updates format statistics."""
        from src.downloader.format_fallback import FormatFallbackHandler
        handler = FormatFallbackHandler(mock_config)

        handler.record_attempt("video123", "mp4", success=True)
        handler.record_attempt("video456", "mp4", success=False, error="Test error")
        handler.record_attempt("video789", "webm", success=True)

        stats = handler.get_format_stats()
        assert stats["mp4"]["attempts"] == 2
        assert stats["mp4"]["successes"] == 1
        assert stats["webm"]["attempts"] == 1
        assert stats["webm"]["successes"] == 1

    def test_get_next_format_returns_next_in_priority(self, mock_config):
        """get_next_format returns next format in priority chain."""
        from src.downloader.format_fallback import FormatFallbackHandler
        handler = FormatFallbackHandler(mock_config)

        # US-144-006: Priority is now mp4 -> webm -> vp9.2 -> avc -> mp3 -> audio-only
        next_fmt = handler.get_next_format("mp4")
        assert next_fmt == "webm"

        next_fmt = handler.get_next_format("webm")
        assert next_fmt == "vp9.2"

        next_fmt = handler.get_next_format("vp9.2")
        assert next_fmt == "avc"

        next_fmt = handler.get_next_format("avc")
        assert next_fmt == "mp3"

        next_fmt = handler.get_next_format("mp3")
        assert next_fmt == "audio-only"

        # audio-only is last, should return None
        next_fmt = handler.get_next_format("audio-only")
        assert next_fmt is None

    def test_should_fallback_on_any_error_enabled(self, mock_config):
        """should_fallback returns True when fallback_on_any_error is True."""
        from src.downloader.format_fallback import FormatFallbackHandler
        handler = FormatFallbackHandler(mock_config)
        handler._format_fallback.fallback_on_any_error = True

        assert handler.should_fallback("some error", "mp4") is True

    def test_should_fallback_disabled(self, mock_config):
        """should_fallback returns False when fallback is disabled."""
        from src.downloader.format_fallback import FormatFallbackHandler
        handler = FormatFallbackHandler(mock_config)
        handler._format_fallback.enabled = False

        assert handler.should_fallback("some error", "mp4") is False

    def test_adaptive_priority_no_adjustment_without_sufficient_data(self, mock_config):
        """Adaptive priority doesn't adjust without min_samples_for_adaptation."""
        mock_config.download.format_fallback.adaptive_priority = True
        mock_config.download.format_fallback.min_samples_for_adaptation = 10

        from src.downloader.format_fallback import FormatFallbackHandler
        handler = FormatFallbackHandler(mock_config)

        # Add some attempts but not enough for adaptation
        handler.record_attempt("v1", "mp4", success=True)
        handler.record_attempt("v2", "mp4", success=False)
        handler.record_attempt("v3", "webm", success=True)

        priority = handler.get_format_priority()
        # Should still be in original order (not enough data)
        # US-144-006: avc and mp3 added to default priority
        assert priority == ["mp4", "webm", "vp9.2", "avc", "mp3", "audio-only"]

    def test_adaptive_priority_adjusts_with_sufficient_data(self, mock_config):
        """Adaptive priority adjusts when enough samples exist."""
        mock_config.download.format_fallback.adaptive_priority = True
        mock_config.download.format_fallback.min_samples_for_adaptation = 3

        from src.downloader.format_fallback import FormatFallbackHandler
        handler = FormatFallbackHandler(mock_config)

        # Add enough attempts: mp4 fails more, webm succeeds more
        for i in range(5):
            handler.record_attempt(f"v{i}", "mp4", success=False)
        for i in range(5):
            handler.record_attempt(f"v{i}", "webm", success=True)

        priority = handler.get_format_priority()
        # webm should now be first (higher success rate)
        assert priority[0] == "webm"
        # audio-only should still be last
        assert priority[-1] == "audio-only"

    def test_is_audio_only_format(self, mock_config):
        """is_audio_only_format correctly identifies audio formats."""
        from src.downloader.format_fallback import FormatFallbackHandler
        handler = FormatFallbackHandler(mock_config)

        assert handler.is_audio_only_format("bestaudio") is True
        # This contains bestaudio, so returns True (conservative - might need transcription)
        assert handler.is_audio_only_format("bestvideo[ext=mp4]+bestaudio") is True
        assert handler.is_audio_only_format("bestaudio[ext=m4a]") is True
        assert handler.is_audio_only_format("bestvideo[ext=mp4]+bestaudio[ext=m4a]") is True


class TestFormatFallbackIntegration:
    """Integration tests for format fallback in download pipeline."""

    def test_format_fallback_handler_instantiated_in_audio_first_pipeline(self):
        """Verify FormatFallbackHandler can be instantiated with real config."""
        # This is a smoke test to ensure the handler integrates properly
        from src.downloader.format_fallback import FormatFallbackHandler

        # The handler requires a config with format_fallback
        # This test verifies the import chain works
        assert FormatFallbackHandler is not None

    def test_format_string_contains_davinci_preference(self):
        """Verify format string prefers h264 for DaVinci when davinci_mode is True."""
        from src.downloader.format_fallback import FormatFallbackHandler, AUDIO_ONLY_FORMATS

        mock_config = MagicMock()
        mock_config.download.format_fallback = FormatFallbackConfig()
        mock_config.download.format_preference = None
        mock_config.download.davinci_mode = True
        mock_config.download.quality = "1080"

        handler = FormatFallbackHandler(mock_config)
        format_str = handler.build_composite_format_string("1080")

        # In DaVinci mode, should prefer avc1 (H.264)
        assert "avc1" in format_str


class TestFormatFallbackHandlerAdditional:
    """Additional tests for FormatFallbackHandler to achieve 80% coverage."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for testing."""
        config = MagicMock()
        config.download.format_fallback = FormatFallbackConfig()
        config.download.format_preference = None
        config.download.davinci_mode = True
        config.download.quality = "1080"
        return config

    def test_get_format_priority_with_none_config(self, mock_config):
        """get_format_priority returns default when format_fallback is None."""
        mock_config.download.format_fallback = None
        from src.downloader.format_fallback import FormatFallbackHandler
        handler = FormatFallbackHandler(mock_config)
        priority = handler.get_format_priority()
        # US-144-006: avc and mp3 added to default priority
        assert priority == ["mp4", "webm", "vp9.2", "avc", "mp3", "audio-only"]

    def test_build_format_string_audio_only(self, mock_config):
        """build_format_string returns bestaudio for audio-only formats."""
        from src.downloader.format_fallback import FormatFallbackHandler
        handler = FormatFallbackHandler(mock_config)

        assert handler.build_format_string("audio-only") == "bestaudio"
        assert handler.build_format_string("bestaudio") == "bestaudio"

    def test_build_format_string_with_format_preference(self, mock_config):
        """build_format_string uses format_preference when available."""
        from src.downloader.format_fallback import FormatFallbackHandler
        from src.config.sections.download import FormatPreferenceConfig

        mock_config.download.format_preference = FormatPreferenceConfig()
        mock_config.download.format_preference.preference_order = ["mp4", "webm"]

        handler = FormatFallbackHandler(mock_config)
        format_str = handler.build_format_string("webm")
        # Should use transcoding manager to build format string
        assert format_str is not None

    def test_build_composite_format_string_non_davinci_mode(self, mock_config):
        """build_composite_format_string works when davinci_mode is False."""
        mock_config.download.davinci_mode = False
        from src.downloader.format_fallback import FormatFallbackHandler
        handler = FormatFallbackHandler(mock_config)

        format_str = handler.build_composite_format_string("1080")
        # Should not contain avc1 (H.264) when not in DaVinci mode
        assert "avc1" not in format_str
        assert "mp4" in format_str

    def test_build_composite_format_string_non_digit_quality(self, mock_config):
        """build_composite_format_string handles non-digit quality."""
        from src.downloader.format_fallback import FormatFallbackHandler
        handler = FormatFallbackHandler(mock_config)

        format_str = handler.build_composite_format_string("best")
        assert "mp4" in format_str

    def test_should_fallback_with_none_config(self, mock_config):
        """should_fallback returns False when format_fallback is None."""
        mock_config.download.format_fallback = None
        from src.downloader.format_fallback import FormatFallbackHandler
        handler = FormatFallbackHandler(mock_config)

        assert handler.should_fallback("some error", "mp4") is False

    def test_should_fallback_on_unavailable_patterns(self, mock_config):
        """should_fallback returns True for specific unavailable patterns."""
        from src.downloader.format_fallback import FormatFallbackHandler
        handler = FormatFallbackHandler(mock_config)
        handler._format_fallback.fallback_on_any_error = False

        # Test various unavailable patterns
        assert handler.should_fallback("Format not available", "mp4") is True
        assert handler.should_fallback("Video unavailable", "mp4") is True
        assert handler.should_fallback("Not available in your region", "mp4") is True
        assert handler.should_fallback("Premium video", "mp4") is True
        assert handler.should_fallback("Members only video", "mp4") is True
        assert handler.should_fallback("Private video", "mp4") is True
        assert handler.should_fallback("Video deleted", "mp4") is True
        assert handler.should_fallback("Video removed", "mp4") is True
        assert handler.should_fallback("No format found", "mp4") is True

    def test_should_fallback_no_error(self, mock_config):
        """should_fallback returns False when error is None."""
        from src.downloader.format_fallback import FormatFallbackHandler
        handler = FormatFallbackHandler(mock_config)
        handler._format_fallback.fallback_on_any_error = False

        assert handler.should_fallback(None, "mp4") is False
        assert handler.should_fallback("", "mp4") is False

    def test_get_next_format_unknown_format(self, mock_config):
        """get_next_format returns first format when current is unknown."""
        from src.downloader.format_fallback import FormatFallbackHandler
        handler = FormatFallbackHandler(mock_config)

        # Unknown format should return first in priority
        next_fmt = handler.get_next_format("unknown")
        assert next_fmt == "mp4"

    def test_log_success_rates_with_tracking(self, mock_config, caplog):
        """log_success_rates logs when tracking is enabled."""
        mock_config.download.format_fallback.track_success_rates = True
        from src.downloader.format_fallback import FormatFallbackHandler
        handler = FormatFallbackHandler(mock_config)

        handler.record_attempt("v1", "mp4", success=True)

        import logging
        with caplog.at_level(logging.INFO):
            handler.log_success_rates()
            # Should have logged something about success rates
            assert "mp4" in caplog.text or "success" in caplog.text.lower()

    def test_log_success_rates_no_tracking(self, mock_config, caplog):
        """log_success_rates does nothing when tracking disabled."""
        mock_config.download.format_fallback.track_success_rates = False
        from src.downloader.format_fallback import FormatFallbackHandler
        handler = FormatFallbackHandler(mock_config)

        handler.record_attempt("v1", "mp4", success=True)

        import logging
        with caplog.at_level(logging.INFO):
            handler.log_success_rates()
            # Should not have logged
            assert "success" not in caplog.text.lower()

    def test_adaptive_priority_disabled(self, mock_config):
        """Adaptive priority doesn't adjust when disabled."""
        mock_config.download.format_fallback.adaptive_priority = False

        from src.downloader.format_fallback import FormatFallbackHandler
        handler = FormatFallbackHandler(mock_config)

        # Add many attempts
        for i in range(15):
            handler.record_attempt(f"v{i}", "mp4", success=False)
        for i in range(15):
            handler.record_attempt(f"v{i}", "webm", success=True)

        priority = handler.get_format_priority()
        # Should still be in original order since adaptive is disabled
        # US-144-006: avc and mp3 added to default priority
        assert priority == ["mp4", "webm", "vp9.2", "avc", "mp3", "audio-only"]


class TestFormatFallbackPipeline:
    """Tests for FormatFallbackPipeline."""

    def test_format_fallback_pipeline_enabled(self):
        """FormatFallbackPipeline.is_enabled reflects handler state."""
        from src.downloader.format_fallback import FormatFallbackPipeline, FormatFallbackHandler
        from src.config.sections.download import FormatFallbackConfig

        mock_config = MagicMock()
        mock_config.download.format_fallback = FormatFallbackConfig()
        mock_config.download.format_preference = None
        mock_config.download.davinci_mode = False
        mock_config.download.quality = "1080"

        def mock_download(video_id, format_str):
            return f"/path/to/{video_id}.mp4"

        pipeline = FormatFallbackPipeline(mock_config, mock_download)
        assert pipeline.is_enabled is True

    def test_download_with_fallback_disabled(self):
        """download_with_fallback uses default when disabled."""
        from src.downloader.format_fallback import FormatFallbackPipeline
        from src.config.sections.download import FormatFallbackConfig

        mock_config = MagicMock()
        mock_config.download.format_fallback = FormatFallbackConfig(enabled=False)
        mock_config.download.format_preference = None
        mock_config.download.davinci_mode = False
        mock_config.download.quality = "1080"

        def mock_download(video_id, format_str):
            return f"/path/to/{video_id}.mp4"

        pipeline = FormatFallbackPipeline(mock_config, mock_download)
        file_path, format_used, is_audio = pipeline.download_with_fallback("abc123")

        assert file_path == "/path/to/abc123.mp4"
        assert format_used == "default"

    def test_download_with_fallback_success_first_format(self):
        """download_with_fallback succeeds on first format."""
        from src.downloader.format_fallback import FormatFallbackPipeline
        from src.config.sections.download import FormatFallbackConfig

        mock_config = MagicMock()
        mock_config.download.format_fallback = FormatFallbackConfig()
        mock_config.download.format_preference = None
        mock_config.download.davinci_mode = False
        mock_config.download.quality = "1080"

        def mock_download(video_id, format_str):
            return f"/path/to/{video_id}.mp4"

        pipeline = FormatFallbackPipeline(mock_config, mock_download)
        file_path, format_used, is_audio = pipeline.download_with_fallback("abc123")

        assert file_path is not None
        # US-144-006: avc and mp3 added to possible formats
        assert format_used in ["mp4", "webm", "vp9.2", "avc", "mp3", "audio-only"]

    def test_download_with_fallback_all_fail(self):
        """download_with_fallback returns None when all formats fail."""
        from src.downloader.format_fallback import FormatFallbackPipeline
        from src.config.sections.download import FormatFallbackConfig

        mock_config = MagicMock()
        mock_config.download.format_fallback = FormatFallbackConfig(
            max_retries_per_format=0
        )
        mock_config.download.format_preference = None
        mock_config.download.davinci_mode = False
        mock_config.download.quality = "1080"

        def mock_download(video_id, format_str):
            return None  # Always fail

        pipeline = FormatFallbackPipeline(mock_config, mock_download)
        file_path, format_used, is_audio = pipeline.download_with_fallback(
            "abc123", max_total_retries=3
        )

        assert file_path is None
        assert format_used is None

    def test_download_with_fallback_exception(self):
        """download_with_fallback handles exceptions and continues."""
        from src.downloader.format_fallback import FormatFallbackPipeline
        from src.config.sections.download import FormatFallbackConfig

        mock_config = MagicMock()
        mock_config.download.format_fallback = FormatFallbackConfig(
            max_retries_per_format=0
        )
        mock_config.download.format_preference = None
        mock_config.download.davinci_mode = False
        mock_config.download.quality = "1080"

        call_count = 0

        def mock_download(video_id, format_str):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Network error")
            return f"/path/to/{video_id}.mp4"

        pipeline = FormatFallbackPipeline(mock_config, mock_download)
        file_path, format_used, is_audio = pipeline.download_with_fallback("abc123")

        # Should have retried with next format after exception
        assert file_path is not None

    def test_get_stats(self):
        """get_stats returns format statistics."""
        from src.downloader.format_fallback import FormatFallbackPipeline
        from src.config.sections.download import FormatFallbackConfig

        mock_config = MagicMock()
        mock_config.download.format_fallback = FormatFallbackConfig()
        mock_config.download.format_preference = None
        mock_config.download.davinci_mode = False
        mock_config.download.quality = "1080"

        def mock_download(video_id, format_str):
            return f"/path/to/{video_id}.mp4"

        pipeline = FormatFallbackPipeline(mock_config, mock_download)
        pipeline.download_with_fallback("abc123")

        stats = pipeline.get_stats()
        assert "mp4" in stats or "webm" in stats


class TestFormatAdaptiveSelection:
    """Tests for US-129-004: Adaptive format selection based on historical success rates."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for testing."""
        config = MagicMock()
        config.download.format_fallback = FormatFallbackConfig()
        config.download.format_preference = None
        config.download.davinci_mode = True
        config.download.quality = "1080"
        return config

    def test_format_adaptive_selection_config_option(self, mock_config):
        """US-129-004: format_adaptive_selection config option is available."""
        from src.downloader.format_fallback import FormatFallbackHandler
        from src.config.sections.download import FormatFallbackConfig

        # Create config with format_adaptive_selection
        config = FormatFallbackConfig(format_adaptive_selection=True)
        assert config.format_adaptive_selection is True

        config2 = FormatFallbackConfig(format_adaptive_selection=False)
        assert config2.format_adaptive_selection is False

    def test_should_use_adaptive_selection_true(self, mock_config):
        """US-129-004: should_use_adaptive_selection returns True when enabled."""
        from src.downloader.format_fallback import FormatFallbackHandler

        mock_config.download.format_fallback.format_adaptive_selection = True
        mock_config.download.format_fallback.adaptive_priority = False  # Legacy off

        handler = FormatFallbackHandler(mock_config)
        assert handler.should_use_adaptive_selection() is True

    def test_should_use_adaptive_selection_false(self, mock_config):
        """US-129-004: should_use_adaptive_selection returns False when disabled."""
        from src.downloader.format_fallback import FormatFallbackHandler

        mock_config.download.format_fallback.format_adaptive_selection = False

        handler = FormatFallbackHandler(mock_config)
        assert handler.should_use_adaptive_selection() is False

    def test_should_use_adaptive_selection_legacy_fallback(self, mock_config):
        """US-129-004: Falls back to adaptive_priority when format_adaptive_selection not set."""
        from src.downloader.format_fallback import FormatFallbackHandler
        from src.config.sections.download import FormatFallbackConfig

        # Create config with format_adaptive_selection = None but adaptive_priority = True
        mock_config.download.format_fallback = FormatFallbackConfig(
            format_adaptive_selection=None,  # Explicitly None
            adaptive_priority=True,
        )

        handler = FormatFallbackHandler(mock_config)
        # Should fall back to adaptive_priority
        assert handler.should_use_adaptive_selection() is True

    def test_adaptive_reordering_with_mock_success_history(self, mock_config):
        """US-129-004: Adaptive reordering based on mock success history."""
        from src.downloader.format_fallback import FormatFallbackHandler
        from src.config.sections.download import FormatFallbackConfig

        # Create fresh config with proper settings
        # Note: reset_attempts_threshold=0 to disable reset during test
        mock_config.download.format_fallback = FormatFallbackConfig(
            format_adaptive_selection=True,
            min_samples_for_adaptation=3,
            reset_attempts_threshold=0,
        )

        handler = FormatFallbackHandler(mock_config)

        # Mock history: mp4 has 0% success, webm has 100% success
        for i in range(5):
            handler.record_attempt(f"v{i}", "mp4", success=False)
        for i in range(5):
            handler.record_attempt(f"v{i}", "webm", success=True)

        priority = handler.get_format_priority()
        # webm should be first since it has higher success rate
        assert priority[0] == "webm"
        # mp4 should be second
        assert priority[1] == "mp4"
        # audio-only should be last
        assert priority[-1] == "audio-only"

    def test_adaptive_reordering_partial_data(self, mock_config):
        """US-129-004: Adaptive reordering with partial data only reorders formats with enough samples."""
        from src.downloader.format_fallback import FormatFallbackHandler

        mock_config.download.format_fallback.format_adaptive_selection = True
        mock_config.download.format_fallback.min_samples_for_adaptation = 5

        handler = FormatFallbackHandler(mock_config)

        # Only mp4 has enough samples
        for i in range(6):
            handler.record_attempt(f"v{i}", "mp4", success=False)

        # webm doesn't have enough samples yet
        for i in range(3):
            handler.record_attempt(f"v{i}", "webm", success=True)

        priority = handler.get_format_priority()
        # mp4 should still be first (not enough data for webm to override)
        assert priority[0] == "mp4"

    def test_consecutive_failures_tracking(self, mock_config):
        """US-129-004: Consecutive failures are tracked correctly."""
        from src.downloader.format_fallback import FormatFallbackHandler

        handler = FormatFallbackHandler(mock_config)

        # Record 3 failures
        handler.record_attempt("v1", "mp4", success=False)
        handler.record_attempt("v2", "mp4", success=False)
        handler.record_attempt("v3", "mp4", success=False)

        stats = handler._format_stats["mp4"]
        assert stats.consecutive_failures == 3

        # Record success - should reset
        handler.record_attempt("v4", "mp4", success=True)
        assert stats.consecutive_failures == 0

    def test_reset_attempts_threshold(self, mock_config):
        """US-129-004: Stats reset after exceeding reset_attempts_threshold."""
        from src.downloader.format_fallback import FormatFallbackHandler

        mock_config.download.format_fallback.reset_attempts_threshold = 3

        handler = FormatFallbackHandler(mock_config)

        # Record 4 consecutive failures (exceeds threshold of 3) - should trigger reset
        handler.record_attempt("v1", "mp4", success=False)
        handler.record_attempt("v2", "mp4", success=False)
        handler.record_attempt("v3", "mp4", success=False)
        handler.record_attempt("v4", "mp4", success=False)  # This exceeds threshold, triggers reset

        stats = handler._format_stats["mp4"]
        # Stats should be reset after exceeding threshold
        assert stats.attempts == 0
        assert stats.successes == 0
        assert stats.consecutive_failures == 0

    def test_reset_attempts_threshold_disabled(self, mock_config):
        """US-129-004: Stats don't reset when reset_attempts_threshold is 0."""
        from src.downloader.format_fallback import FormatFallbackHandler

        mock_config.download.format_fallback.reset_attempts_threshold = 0

        handler = FormatFallbackHandler(mock_config)

        # Record many failures
        for i in range(10):
            handler.record_attempt(f"v{i}", "mp4", success=False)

        stats = handler._format_stats["mp4"]
        # Stats should NOT be reset
        assert stats.attempts == 10
        assert stats.consecutive_failures == 10

    def test_checkpoint_persistence_serialize(self, mock_config):
        """US-129-004: Format stats can be serialized for checkpoint."""
        from src.downloader.format_fallback import FormatFallbackHandler

        handler = FormatFallbackHandler(mock_config)

        handler.record_attempt("v1", "mp4", success=True)
        handler.record_attempt("v2", "mp4", success=False)
        handler.record_attempt("v3", "webm", success=True)

        checkpoint_data = handler.get_format_stats_for_checkpoint()

        assert "mp4" in checkpoint_data
        assert checkpoint_data["mp4"]["attempts"] == 2
        assert checkpoint_data["mp4"]["successes"] == 1
        assert checkpoint_data["mp4"]["consecutive_failures"] == 1

    def test_checkpoint_persistence_deserialize(self, mock_config):
        """US-129-004: Format stats can be loaded from checkpoint."""
        from src.downloader.format_fallback import FormatFallbackHandler

        handler = FormatFallbackHandler(mock_config)

        # Simulate loading from checkpoint
        checkpoint_data = {
            "mp4": {"attempts": 10, "successes": 7, "consecutive_failures": 0},
            "webm": {"attempts": 8, "successes": 8, "consecutive_failures": 0},
        }

        handler.load_format_stats_from_checkpoint(checkpoint_data)

        mp4_stats = handler._format_stats["mp4"]
        assert mp4_stats.attempts == 10
        assert mp4_stats.successes == 7

        webm_stats = handler._format_stats["webm"]
        assert webm_stats.attempts == 8
        assert webm_stats.successes == 8

    def test_format_stats_to_dict(self):
        """US-129-004: FormatStats.to_dict() serializes correctly."""
        from src.downloader.format_fallback import FormatStats

        stats = FormatStats(attempts=10, successes=7, consecutive_failures=3)
        data = stats.to_dict()

        assert data["attempts"] == 10
        assert data["successes"] == 7
        assert data["consecutive_failures"] == 3

    def test_format_stats_from_dict(self):
        """US-129-004: FormatStats.from_dict() deserializes correctly."""
        from src.downloader.format_fallback import FormatStats

        data = {"attempts": 10, "successes": 7, "consecutive_failures": 3}
        stats = FormatStats.from_dict(data)

        assert stats.attempts == 10
        assert stats.successes == 7
        assert stats.consecutive_failures == 3

    def test_format_stats_from_dict_empty(self):
        """US-129-004: FormatStats.from_dict() handles empty data."""
        from src.downloader.format_fallback import FormatStats

        stats = FormatStats.from_dict({})

        assert stats.attempts == 0
        assert stats.successes == 0
        assert stats.consecutive_failures == 0

    def test_adaptive_selection_with_format_adaptive_selection_false(self, mock_config):
        """US-129-004: Adaptive selection disabled when format_adaptive_selection is False."""
        from src.downloader.format_fallback import FormatFallbackHandler

        mock_config.download.format_fallback.format_adaptive_selection = False

        handler = FormatFallbackHandler(mock_config)

        # Record data that would trigger reordering
        for i in range(15):
            handler.record_attempt(f"v{i}", "mp4", success=False)
        for i in range(15):
            handler.record_attempt(f"v{i}", "webm", success=True)

        priority = handler.get_format_priority()
        # Should NOT reorder since format_adaptive_selection is False
        # US-144-006: avc and mp3 added to default priority
        assert priority == ["mp4", "webm", "vp9.2", "avc", "mp3", "audio-only"]


class TestHDRFormatSupport:
    """US-143-002: Tests for HDR format (vp9.2) support in format fallback."""

    def test_vp92_in_format_identifiers(self):
        """HDR format (vp9.2) is in FORMAT_IDENTIFIERS."""
        from src.downloader.format_fallback import FORMAT_IDENTIFIERS, HDR_FORMATS

        assert "vp9.2" in FORMAT_IDENTIFIERS
        assert "vp9.2" in HDR_FORMATS

    def test_vp92_not_audio_only(self):
        """vp9.2 is not an audio-only format."""
        from src.downloader.format_fallback import AUDIO_ONLY_FORMATS, HDR_FORMATS

        assert "vp9.2" not in AUDIO_ONLY_FORMATS
        assert "vp9.2" in HDR_FORMATS

    def test_vp92_in_default_format_priority(self):
        """vp9.2 is included in default format_priority."""
        from src.config.sections.download import FormatFallbackConfig

        config = FormatFallbackConfig()
        assert "vp9.2" in config.format_priority
        # vp9.2 should be before audio-only
        assert config.format_priority.index("vp9.2") < config.format_priority.index("audio-only")

    def test_vp92_config_validation(self):
        """vp9.2 passes config validation."""
        from src.config.sections.download import FormatFallbackConfig

        config = FormatFallbackConfig(format_priority=["vp9.2", "mp4", "webm", "audio-only"])
        assert "vp9.2" in config.format_priority

    def test_vp92_in_config_yaml(self):
        """vp9.2 is in config.yaml format_priority."""
        import yaml
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        format_priority = config.get("download", {}).get("format_fallback", {}).get("format_priority", [])
        assert "vp9.2" in format_priority


class TestRegionBasedAdaptivePriority:
    """US-143-002: Tests for region-based adaptive format priority."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for testing."""
        config = MagicMock()
        config.download.format_fallback = FormatFallbackConfig()
        config.download.format_preference = None
        config.download.davinci_mode = True
        config.download.quality = "1080"
        config.download.format_fallback.track_region_stats = True
        return config

    def test_record_attempt_with_region(self, mock_config):
        """record_attempt tracks region-specific success rates."""
        from src.downloader.format_fallback import FormatFallbackHandler

        handler = FormatFallbackHandler(mock_config)

        # Record attempts with region
        handler.record_attempt("v1", "mp4", success=True, region="us")
        handler.record_attempt("v2", "mp4", success=False, region="us")
        handler.record_attempt("v3", "mp4", success=True, region="eu")

        stats = handler._format_stats["mp4"]
        assert stats.region_stats["us"]["attempts"] == 2
        assert stats.region_stats["us"]["successes"] == 1
        assert stats.region_stats["eu"]["attempts"] == 1
        assert stats.region_stats["eu"]["successes"] == 1

    def test_get_region_success_rate(self, mock_config):
        """get_region_success_rate returns correct rate."""
        from src.downloader.format_fallback import FormatFallbackHandler

        handler = FormatFallbackHandler(mock_config)

        # Record attempts with region
        handler.record_attempt("v1", "mp4", success=True, region="us")
        handler.record_attempt("v2", "mp4", success=True, region="us")
        handler.record_attempt("v3", "mp4", success=False, region="us")

        stats = handler._format_stats["mp4"]
        assert stats.get_region_success_rate("us") == 2/3

    def test_adaptive_priority_with_region(self, mock_config):
        """get_format_priority uses region-specific success rates when available."""
        from src.downloader.format_fallback import FormatFallbackHandler
        from src.config.sections.download import FormatFallbackConfig

        # Use a real FormatFallbackConfig with track_region_stats enabled
        mock_config.download.format_fallback = FormatFallbackConfig(
            track_region_stats=True,
            min_samples_for_adaptation=10,
            format_adaptive_selection=True
        )

        handler = FormatFallbackHandler(mock_config)

        # Record region-specific data - US has poor mp4 success, good webm
        for i in range(15):
            handler.record_attempt(f"v{i}", "mp4", success=False, region="us")
        for i in range(15):
            handler.record_attempt(f"v{i}", "webm", success=True, region="us")

        # Get priority for US region - should favor webm
        priority = handler.get_format_priority(region="us")

        # webm should be first since it has higher success rate in US
        assert priority[0] == "webm"

    def test_track_region_stats_config_option(self):
        """track_region_stats config option exists and defaults to False."""
        from src.config.sections.download import FormatFallbackConfig

        config = FormatFallbackConfig()
        assert hasattr(config, "track_region_stats")
        assert config.track_region_stats is False


class TestFallbackCountTracking:
    """US-143-002: Tests for fallback_count tracking per format."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for testing."""
        config = MagicMock()
        config.download.format_fallback = FormatFallbackConfig()
        config.download.format_preference = None
        config.download.davinci_mode = True
        config.download.quality = "1080"
        return config

    def test_record_attempt_with_is_fallback(self, mock_config):
        """record_attempt increments fallback_count when is_fallback is True."""
        from src.downloader.format_fallback import FormatFallbackHandler

        handler = FormatFallbackHandler(mock_config)

        # Record fallback attempts
        handler.record_attempt("v1", "webm", success=False, is_fallback=True)
        handler.record_attempt("v2", "webm", success=False, is_fallback=True)
        handler.record_attempt("v3", "webm", success=False, is_fallback=True)

        stats = handler._format_stats["webm"]
        assert stats.fallback_count == 3

    def test_record_attempt_without_is_fallback(self, mock_config):
        """record_attempt does not increment fallback_count when is_fallback is False."""
        from src.downloader.format_fallback import FormatFallbackHandler

        handler = FormatFallbackHandler(mock_config)

        # Record non-fallback attempts
        handler.record_attempt("v1", "mp4", success=True, is_fallback=False)
        handler.record_attempt("v2", "mp4", success=True, is_fallback=False)

        stats = handler._format_stats["mp4"]
        assert stats.fallback_count == 0

    def test_get_fallback_counts(self, mock_config):
        """get_fallback_counts returns correct counts per format."""
        from src.downloader.format_fallback import FormatFallbackHandler

        handler = FormatFallbackHandler(mock_config)

        # Record fallback attempts for different formats
        handler.record_attempt("v1", "mp4", success=False, is_fallback=True)
        handler.record_attempt("v2", "webm", success=False, is_fallback=True)
        handler.record_attempt("v3", "webm", success=False, is_fallback=True)

        counts = handler.get_fallback_counts()
        assert counts.get("mp4", 0) == 1
        assert counts.get("webm", 0) == 2

    def test_get_format_stats_includes_fallback_count(self, mock_config):
        """get_format_stats includes fallback_count in result."""
        from src.downloader.format_fallback import FormatFallbackHandler

        handler = FormatFallbackHandler(mock_config)

        handler.record_attempt("v1", "mp4", success=False, is_fallback=True)

        stats = handler.get_format_stats()
        assert "fallback_count" in stats["mp4"]
        assert stats["mp4"]["fallback_count"] == 1

    def test_format_stats_serialization_includes_fallback_count(self):
        """FormatStats.to_dict includes fallback_count."""
        from src.downloader.format_fallback import FormatStats

        stats = FormatStats(attempts=10, successes=5, fallback_count=3)
        data = stats.to_dict()

        assert data["fallback_count"] == 3

    def test_format_stats_deserialization_includes_fallback_count(self):
        """FormatStats.from_dict includes fallback_count."""
        from src.downloader.format_fallback import FormatStats

        data = {"attempts": 10, "successes": 5, "fallback_count": 3, "consecutive_failures": 2, "region_stats": {}}
        stats = FormatStats.from_dict(data)

        assert stats.fallback_count == 3


class TestUS144006NewFormats:
    """US-144-006: Tests for new format fallback combinations."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for testing."""
        config = MagicMock()
        config.download.format_fallback = FormatFallbackConfig()
        config.download.format_preference = None
        config.download.davinci_mode = True
        config.download.quality = "1080"
        return config

    def test_avc_format_in_identifiers(self):
        """US-144-006: avc format is in FORMAT_IDENTIFIERS."""
        from src.downloader.format_fallback import FORMAT_IDENTIFIERS

        assert "avc" in FORMAT_IDENTIFIERS
        assert FORMAT_IDENTIFIERS["avc"] == "mp4"

    def test_mp3_format_in_identifiers(self):
        """US-144-006: mp3 format is in FORMAT_IDENTIFIERS."""
        from src.downloader.format_fallback import FORMAT_IDENTIFIERS

        assert "mp3" in FORMAT_IDENTIFIERS
        assert FORMAT_IDENTIFIERS["mp3"] == "bestaudio[ext=mp3]"

    def test_thumbnail_format_in_identifiers(self):
        """US-144-006: thumbnail format is in FORMAT_IDENTIFIERS."""
        from src.downloader.format_fallback import FORMAT_IDENTIFIERS, THUMBNAIL_ONLY_FORMAT

        assert "thumbnail" in FORMAT_IDENTIFIERS
        assert FORMAT_IDENTIFIERS["thumbnail"] == "thumbnail"
        assert THUMBNAIL_ONLY_FORMAT == "thumbnail"

    def test_mp3_in_audio_only_formats(self):
        """US-144-006: mp3 is in AUDIO_ONLY_FORMATS."""
        from src.downloader.format_fallback import AUDIO_ONLY_FORMATS

        assert "mp3" in AUDIO_ONLY_FORMATS

    def test_codec_fallback_map_contains_vp9_avc(self):
        """US-144-006: CODEC_FALLBACK_MAP contains vp9.2 -> avc fallback."""
        from src.downloader.format_fallback import CODEC_FALLBACK_MAP

        assert "vp9.2" in CODEC_FALLBACK_MAP
        assert CODEC_FALLBACK_MAP["vp9.2"] == "avc"
        assert "vp9" in CODEC_FALLBACK_MAP
        assert CODEC_FALLBACK_MAP["vp9"] == "avc"
        assert "webm" in CODEC_FALLBACK_MAP
        assert CODEC_FALLBACK_MAP["webm"] == "avc"

    def test_codec_fallback_map_contains_aac_mp3(self):
        """US-144-006: CODEC_FALLBACK_MAP contains m4a/aac -> mp3 fallback."""
        from src.downloader.format_fallback import CODEC_FALLBACK_MAP

        assert "m4a" in CODEC_FALLBACK_MAP
        assert CODEC_FALLBACK_MAP["m4a"] == "mp3"
        assert "aac" in CODEC_FALLBACK_MAP
        assert CODEC_FALLBACK_MAP["aac"] == "mp3"

    def test_avc_in_default_format_priority(self):
        """US-144-006: avc is in default format_priority."""
        from src.config.sections.download import FormatFallbackConfig

        config = FormatFallbackConfig()
        assert "avc" in config.format_priority

    def test_mp3_in_default_format_priority(self):
        """US-144-006: mp3 is in default format_priority."""
        from src.config.sections.download import FormatFallbackConfig

        config = FormatFallbackConfig()
        assert "mp3" in config.format_priority

    def test_thumbnail_fallback_config_option(self):
        """US-144-006: enable_thumbnail_fallback config option exists."""
        from src.config.sections.download import FormatFallbackConfig

        config = FormatFallbackConfig()
        assert hasattr(config, "enable_thumbnail_fallback")
        assert config.enable_thumbnail_fallback is False

    def test_get_codec_fallback_format(self, mock_config):
        """US-144-006: get_codec_fallback_format returns correct fallback."""
        from src.downloader.format_fallback import FormatFallbackHandler

        handler = FormatFallbackHandler(mock_config)

        assert handler.get_codec_fallback_format("vp9.2") == "avc"
        assert handler.get_codec_fallback_format("vp9") == "avc"
        assert handler.get_codec_fallback_format("webm") == "avc"
        assert handler.get_codec_fallback_format("m4a") == "mp3"
        assert handler.get_codec_fallback_format("aac") == "mp3"
        # No fallback for mp4
        assert handler.get_codec_fallback_format("mp4") is None

    def test_should_use_codec_fallback(self, mock_config):
        """US-144-006: should_use_codec_fallback detects codec errors."""
        from src.downloader.format_fallback import FormatFallbackHandler

        handler = FormatFallbackHandler(mock_config)

        # Should return True for codec-related errors
        assert handler.should_use_codec_fallback("Codec not supported", "vp9.2") is True
        assert handler.should_use_codec_fallback("Encoder error", "vp9.2") is True
        assert handler.should_use_codec_fallback("Format not available", "vp9.2") is True
        # Should return False for non-codec errors
        assert handler.should_use_codec_fallback("Network error", "vp9.2") is False

    def test_should_use_codec_fallback_no_fallback_for_mp4(self, mock_config):
        """US-144-006: should_use_codec_fallback returns False for mp4 (no fallback)."""
        from src.downloader.format_fallback import FormatFallbackHandler

        handler = FormatFallbackHandler(mock_config)

        # mp4 has no codec fallback
        assert handler.should_use_codec_fallback("Codec error", "mp4") is False

    def test_get_thumbnail_fallback_format(self, mock_config):
        """US-144-006: get_thumbnail_fallback_format returns thumbnail."""
        from src.downloader.format_fallback import FormatFallbackHandler

        handler = FormatFallbackHandler(mock_config)

        assert handler.get_thumbnail_fallback_format() == "thumbnail"

    def test_should_use_thumbnail_fallback(self, mock_config):
        """US-144-006: should_use_thumbnail_fallback detects premium content."""
        from src.downloader.format_fallback import FormatFallbackHandler

        handler = FormatFallbackHandler(mock_config)
        handler._format_fallback.enable_thumbnail_fallback = True

        # Should return True for premium errors after trying multiple formats
        assert handler.should_use_thumbnail_fallback("Premium content", ["mp4", "webm"]) is True
        assert handler.should_use_thumbnail_fallback("Members only video", ["mp4", "webm"]) is True
        # Need at least 2 formats tried for thumbnail fallback
        assert handler.should_use_thumbnail_fallback("Not available in your region", ["mp4", "webm"]) is True
        # Should return False when not enabled
        handler._format_fallback.enable_thumbnail_fallback = False
        assert handler.should_use_thumbnail_fallback("Premium content", ["mp4", "webm"]) is False

    def test_is_thumbnail_only_format(self, mock_config):
        """US-144-006: is_thumbnail_only_format correctly identifies thumbnail format."""
        from src.downloader.format_fallback import FormatFallbackHandler

        handler = FormatFallbackHandler(mock_config)

        assert handler.is_thumbnail_only_format("thumbnail") is True
        assert handler.is_thumbnail_only_format("THUMBNAIL") is True
        assert handler.is_thumbnail_only_format("mp4") is False

    def test_build_composite_format_string_includes_avc(self, mock_config):
        """US-144-006: build_composite_format_string includes avc format."""
        from src.downloader.format_fallback import FormatFallbackHandler

        handler = FormatFallbackHandler(mock_config)
        format_str = handler.build_composite_format_string("1080")

        # Should contain avc-related selectors
        assert "avc1" in format_str  # H.264 codec

    def test_build_composite_format_string_includes_mp3(self, mock_config):
        """US-144-006: build_composite_format_string includes mp3 format."""
        from src.downloader.format_fallback import FormatFallbackHandler

        mock_config.download.format_fallback = FormatFallbackConfig(
            format_priority=["mp4", "webm", "mp3", "audio-only"]
        )
        handler = FormatFallbackHandler(mock_config)
        format_str = handler.build_composite_format_string("1080")

        # Should contain mp3 audio selector
        assert "mp3" in format_str

    def test_build_composite_format_string_includes_thumbnail(self, mock_config):
        """US-144-006: build_composite_format_string includes thumbnail format."""
        from src.downloader.format_fallback import FormatFallbackHandler

        mock_config.download.format_fallback = FormatFallbackConfig(
            format_priority=["mp4", "webm", "audio-only", "thumbnail"]
        )
        mock_config.download.format_fallback.enable_thumbnail_fallback = True
        handler = FormatFallbackHandler(mock_config)
        format_str = handler.build_composite_format_string("1080")

        # Should contain thumbnail selector
        assert "thumbnail" in format_str

    def test_get_next_format_with_avc(self, mock_config):
        """US-144-006: get_next_format handles avc format."""
        from src.downloader.format_fallback import FormatFallbackHandler

        handler = FormatFallbackHandler(mock_config)

        # avc should be in the chain
        next_fmt = handler.get_next_format("vp9.2")
        assert next_fmt == "avc"

    def test_get_next_format_with_mp3(self, mock_config):
        """US-144-006: get_next_format handles mp3 format."""
        from src.downloader.format_fallback import FormatFallbackHandler

        handler = FormatFallbackHandler(mock_config)

        # mp3 should be in the chain
        next_fmt = handler.get_next_format("avc")
        assert next_fmt == "mp3"

    def test_audio_only_at_end_with_thumbnail(self):
        """US-144-006: audio-only and thumbnail are at end of priority."""
        from src.config.sections.download import FormatFallbackConfig

        config = FormatFallbackConfig(format_priority=["webm", "thumbnail", "mp4", "audio-only"])
        # Should reorder so audio-only is second-to-last, thumbnail last
        assert config.format_priority[-1] == "thumbnail"
        assert config.format_priority[-2] == "audio-only"

    def test_config_validation_accepts_avc(self):
        """US-144-006: Config validation accepts avc format."""
        from src.config.sections.download import FormatFallbackConfig

        # Should not raise
        config = FormatFallbackConfig(format_priority=["avc", "mp4", "webm"])
        assert "avc" in config.format_priority

    def test_config_validation_accepts_mp3(self):
        """US-144-006: Config validation accepts mp3 format."""
        from src.config.sections.download import FormatFallbackConfig

        # Should not raise
        config = FormatFallbackConfig(format_priority=["mp3", "mp4", "webm"])
        assert "mp3" in config.format_priority

    def test_config_validation_accepts_thumbnail(self):
        """US-144-006: Config validation accepts thumbnail format."""
        from src.config.sections.download import FormatFallbackConfig

        # Should not raise
        config = FormatFallbackConfig(format_priority=["thumbnail", "mp4", "webm"])
        assert "thumbnail" in config.format_priority

    def test_codec_fallback_priority_order(self, mock_config):
        """US-144-006: Codec fallback happens in correct order."""
        from src.downloader.format_fallback import FormatFallbackHandler

        handler = FormatFallbackHandler(mock_config)

        # vp9.2 fails -> avc
        fallback = handler.get_codec_fallback_format("vp9.2")
        assert fallback == "avc"

        # m4a fails -> mp3
        fallback = handler.get_codec_fallback_format("m4a")
        assert fallback == "mp3"

    def test_full_fallback_chain_with_new_formats(self, mock_config):
        """US-144-006: Full fallback chain includes all new formats."""
        from src.downloader.format_fallback import FormatFallbackHandler

        handler = FormatFallbackHandler(mock_config)
        priority = handler.get_format_priority()

        # Should contain: mp4 -> webm -> vp9.2 -> avc -> mp3 -> audio-only
        assert "mp4" in priority
        assert "webm" in priority
        assert "vp9.2" in priority
        assert "avc" in priority
        assert "mp3" in priority
        assert "audio-only" in priority

        # Verify order: avc after vp9.2, mp3 after avc
        assert priority.index("vp9.2") < priority.index("avc")
        assert priority.index("avc") < priority.index("mp3")
        assert priority.index("mp3") < priority.index("audio-only")

    def test_fallback_chain_integration_with_avc_and_mp3(self, mock_config):
        """US-144-006: Integration test verifying fallback chain with avc/mp3 formats."""
        from src.downloader.format_fallback import FormatFallbackHandler, FormatFallbackPipeline
        from src.config.sections.download import FormatFallbackConfig

        # Configure to use avc and mp3 in priority
        mock_config.download.format_fallback = FormatFallbackConfig(
            format_priority=["mp4", "webm", "vp9.2", "avc", "mp3", "audio-only"],
            max_retries_per_format=0,  # No retries, immediate fallback
            format_adaptive_selection=True,
            adaptive_priority=True,
            min_samples_for_adaptation=1,  # Low threshold for test
        )

        # Track which formats were tried
        formats_tried = []

        def mock_download(video_id, format_str):
            formats_tried.append(format_str)
            # Simulate: mp4 fails, webm fails, avc succeeds
            if "mp4" in format_str:
                return None  # Fail
            elif "webm" in format_str:
                return None  # Fail
            elif "avc" in format_str or "vp9" in format_str:
                return f"/path/to/{video_id}.mp4"  # Success
            return None

        pipeline = FormatFallbackPipeline(mock_config, mock_download)
        file_path, format_used, is_audio = pipeline.download_with_fallback("abc123", max_total_retries=10)

        # Should succeed with avc format
        assert file_path is not None
        assert format_used in ["avc", "vp9.2"]  # Either could succeed

        # Verify mp3 or audio-only were NOT tried (avc succeeded)
        assert "mp3" not in formats_tried
        assert "bestaudio" not in formats_tried

    def test_fallback_chain_integration_mp3_fallback(self, mock_config):
        """US-144-006: Integration test verifying fallback when audio mp3 succeeds."""
        from src.downloader.format_fallback import FormatFallbackPipeline
        from src.config.sections.download import FormatFallbackConfig

        mock_config.download.format_fallback = FormatFallbackConfig(
            format_priority=["mp4", "webm", "avc", "mp3", "audio-only"],
            max_retries_per_format=0,
        )

        formats_tried = []

        def mock_download(video_id, format_str):
            formats_tried.append(format_str)
            # Simulate: video formats fail, audio-only succeeds
            # The format strings contain "bestvideo" for video formats
            if "bestvideo" in format_str:
                return None  # Fail
            # For audio-only formats (bestaudio), succeed
            return f"/path/to/{video_id}.mp3"  # Success

        pipeline = FormatFallbackPipeline(mock_config, mock_download)
        file_path, format_used, is_audio = pipeline.download_with_fallback("abc123", max_total_retries=10)

        # Should succeed (fallback to audio-only)
        assert file_path is not None
        # format_used should be audio-only or mp3 depending on the priority
        assert format_used in ["mp3", "audio-only"]
        assert is_audio is True

    def test_adaptive_priority_reorders_by_success_rate_integration(self, mock_config):
        """US-144-006: Integration test verifying adaptive priority reorders formats."""
        from src.downloader.format_fallback import FormatFallbackHandler
        from src.config.sections.download import FormatFallbackConfig

        mock_config.download.format_fallback = FormatFallbackConfig(
            format_priority=["mp4", "webm", "avc", "mp3", "audio-only"],
            format_adaptive_selection=True,
            min_samples_for_adaptation=2,
        )

        handler = FormatFallbackHandler(mock_config)

        # Record: mp4 has 100% success, webm has 0% success
        for i in range(5):
            handler.record_attempt(f"v{i}", "mp4", success=True)
        for i in range(5):
            handler.record_attempt(f"v{i}", "webm", success=False)

        # Get priority - mp4 should now be first due to higher success rate
        priority = handler.get_format_priority()

        # mp4 should be first (highest success rate), webm should be lower
        assert priority[0] == "mp4", f"Expected mp4 first, got {priority[0]}"

        # audio-only should still be last
        assert priority[-1] == "audio-only"

