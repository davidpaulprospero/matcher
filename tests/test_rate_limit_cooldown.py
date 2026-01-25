"""
Tests for rate limit cooldown tracking across sessions (US-008).

Verifies:
- Rate limit events saved to checkpoint with timestamp
- Cooldown check on resume detects rate limit within cooldown period
- Recovery mode applies longer delays and faster escalation
- Cooldown status logged on pipeline resume
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

from src.downloader.checkpoint import DownloadCheckpoint
from src.config.sections.download import RateLimitConfig


# =============================================================================
# Test RateLimitConfig with resume_cooldown_minutes
# =============================================================================

class TestRateLimitConfig:
    """Tests for RateLimitConfig dataclass."""

    def test_default_resume_cooldown_minutes(self):
        """Default resume_cooldown_minutes should be 15."""
        config = RateLimitConfig()
        assert config.resume_cooldown_minutes == 15.0

    def test_custom_resume_cooldown_minutes(self):
        """Custom resume_cooldown_minutes should be respected."""
        config = RateLimitConfig(resume_cooldown_minutes=30.0)
        assert config.resume_cooldown_minutes == 30.0

    def test_all_config_values(self):
        """Verify all config values can be set."""
        config = RateLimitConfig(
            initial_backoff_seconds=10.0,
            max_backoff_before_rotate=120.0,
            backoff_multiplier=3.0,
            resume_cooldown_minutes=20.0
        )
        assert config.initial_backoff_seconds == 10.0
        assert config.max_backoff_before_rotate == 120.0
        assert config.backoff_multiplier == 3.0
        assert config.resume_cooldown_minutes == 20.0


# =============================================================================
# Test DownloadCheckpoint with rate limit tracking fields
# =============================================================================

class TestDownloadCheckpointRateLimitFields:
    """Tests for rate limit tracking fields in DownloadCheckpoint."""

    def test_default_rate_limit_fields(self):
        """New checkpoint should have None/0 rate limit fields."""
        checkpoint = DownloadCheckpoint(
            completed_keywords=[],
            completed_videos=[],
            failed_keywords=[],
            current_keyword=None,
            current_tier=None,
            timestamp=datetime.now().isoformat()
        )
        assert checkpoint.last_rate_limit_timestamp is None
        assert checkpoint.rate_limit_event_count == 0

    def test_rate_limit_fields_in_to_dict(self):
        """Rate limit fields should be included in to_dict output."""
        timestamp = datetime.now().isoformat()
        checkpoint = DownloadCheckpoint(
            completed_keywords=["kw1"],
            completed_videos=[],
            failed_keywords=[],
            current_keyword=None,
            current_tier=None,
            timestamp=timestamp,
            last_rate_limit_timestamp=timestamp,
            rate_limit_event_count=5
        )
        data = checkpoint.to_dict()
        assert data['last_rate_limit_timestamp'] == timestamp
        assert data['rate_limit_event_count'] == 5

    def test_from_dict_with_rate_limit_fields(self):
        """from_dict should restore rate limit fields."""
        timestamp = datetime.now().isoformat()
        data = {
            'completed_keywords': [],
            'completed_videos': [],
            'failed_keywords': [],
            'current_keyword': None,
            'current_tier': None,
            'timestamp': timestamp,
            'last_rate_limit_timestamp': timestamp,
            'rate_limit_event_count': 3
        }
        checkpoint = DownloadCheckpoint.from_dict(data)
        assert checkpoint.last_rate_limit_timestamp == timestamp
        assert checkpoint.rate_limit_event_count == 3

    def test_from_dict_without_rate_limit_fields_backward_compat(self):
        """from_dict should handle old checkpoints without rate limit fields."""
        data = {
            'completed_keywords': [],
            'completed_videos': [],
            'failed_keywords': [],
            'current_keyword': None,
            'current_tier': None,
            'timestamp': datetime.now().isoformat()
        }
        checkpoint = DownloadCheckpoint.from_dict(data)
        assert checkpoint.last_rate_limit_timestamp is None
        assert checkpoint.rate_limit_event_count == 0


# =============================================================================
# Test VideoDownloader cooldown checking
# =============================================================================

class TestCooldownCheck:
    """Tests for _check_rate_limit_cooldown method."""

    @pytest.fixture
    def mock_downloader(self):
        """Create a mock VideoDownloader with rate limit config."""
        with patch('src.downloader.core.VideoDownloader.__init__', return_value=None):
            from src.downloader.core import VideoDownloader
            downloader = VideoDownloader.__new__(VideoDownloader)

            # Set up minimal required attributes
            downloader._rate_limit_backoff_count = 0
            downloader._rate_limit_total_delay = 0.0
            downloader._rate_limit_event_count = 0
            downloader._in_cooldown_recovery_mode = False

            # Mock download_config with rate_limit settings
            rate_limit_config = MagicMock()
            rate_limit_config.resume_cooldown_minutes = 15.0

            download_config = MagicMock()
            download_config.rate_limit = rate_limit_config
            downloader.download_config = download_config

            return downloader

    def test_no_previous_rate_limit(self, mock_downloader):
        """No cooldown if checkpoint has no rate limit timestamp."""
        checkpoint = DownloadCheckpoint(
            completed_keywords=[],
            completed_videos=[],
            failed_keywords=[],
            current_keyword=None,
            current_tier=None,
            timestamp=datetime.now().isoformat(),
            last_rate_limit_timestamp=None,
            rate_limit_event_count=0
        )

        result = mock_downloader._check_rate_limit_cooldown(checkpoint)
        assert result is False

    def test_within_cooldown_period(self, mock_downloader):
        """Should return True if rate limit was within cooldown period."""
        # Rate limit 5 minutes ago, cooldown is 15 minutes
        five_minutes_ago = datetime.now() - timedelta(minutes=5)
        checkpoint = DownloadCheckpoint(
            completed_keywords=[],
            completed_videos=[],
            failed_keywords=[],
            current_keyword=None,
            current_tier=None,
            timestamp=datetime.now().isoformat(),
            last_rate_limit_timestamp=five_minutes_ago.isoformat(),
            rate_limit_event_count=3
        )

        result = mock_downloader._check_rate_limit_cooldown(checkpoint)
        assert result is True

    def test_outside_cooldown_period(self, mock_downloader):
        """Should return False if rate limit was outside cooldown period."""
        # Rate limit 20 minutes ago, cooldown is 15 minutes
        twenty_minutes_ago = datetime.now() - timedelta(minutes=20)
        checkpoint = DownloadCheckpoint(
            completed_keywords=[],
            completed_videos=[],
            failed_keywords=[],
            current_keyword=None,
            current_tier=None,
            timestamp=datetime.now().isoformat(),
            last_rate_limit_timestamp=twenty_minutes_ago.isoformat(),
            rate_limit_event_count=10
        )

        result = mock_downloader._check_rate_limit_cooldown(checkpoint)
        assert result is False

    def test_invalid_timestamp_ignored(self, mock_downloader):
        """Invalid timestamp should be ignored, return False."""
        checkpoint = DownloadCheckpoint(
            completed_keywords=[],
            completed_videos=[],
            failed_keywords=[],
            current_keyword=None,
            current_tier=None,
            timestamp=datetime.now().isoformat(),
            last_rate_limit_timestamp="invalid-timestamp",
            rate_limit_event_count=5
        )

        result = mock_downloader._check_rate_limit_cooldown(checkpoint)
        assert result is False

    def test_custom_cooldown_period(self, mock_downloader):
        """Custom cooldown period should be respected."""
        # Set custom cooldown to 5 minutes
        mock_downloader.download_config.rate_limit.resume_cooldown_minutes = 5.0

        # Rate limit 3 minutes ago
        three_minutes_ago = datetime.now() - timedelta(minutes=3)
        checkpoint = DownloadCheckpoint(
            completed_keywords=[],
            completed_videos=[],
            failed_keywords=[],
            current_keyword=None,
            current_tier=None,
            timestamp=datetime.now().isoformat(),
            last_rate_limit_timestamp=three_minutes_ago.isoformat(),
            rate_limit_event_count=2
        )

        # Should be within cooldown (3 < 5)
        result = mock_downloader._check_rate_limit_cooldown(checkpoint)
        assert result is True

        # Now set cooldown to 2 minutes
        mock_downloader.download_config.rate_limit.resume_cooldown_minutes = 2.0

        # Should be outside cooldown (3 > 2)
        result = mock_downloader._check_rate_limit_cooldown(checkpoint)
        assert result is False


# =============================================================================
# Test rate limit event recording
# =============================================================================

class TestRateLimitEventRecording:
    """Tests for rate limit event recording."""

    @pytest.fixture
    def mock_downloader_with_checkpoint(self):
        """Create a mock VideoDownloader with checkpoint."""
        with patch('src.downloader.core.VideoDownloader.__init__', return_value=None):
            from src.downloader.core import VideoDownloader
            downloader = VideoDownloader.__new__(VideoDownloader)

            # Set up minimal required attributes
            downloader._rate_limit_backoff_count = 0
            downloader._rate_limit_total_delay = 0.0
            downloader._rate_limit_event_count = 0
            downloader._in_cooldown_recovery_mode = False

            # Create checkpoint
            downloader.checkpoint = DownloadCheckpoint(
                completed_keywords=[],
                completed_videos=[],
                failed_keywords=[],
                current_keyword=None,
                current_tier=None,
                timestamp=datetime.now().isoformat(),
                last_rate_limit_timestamp=None,
                rate_limit_event_count=0
            )

            # Mock _save_checkpoint
            downloader._save_checkpoint = MagicMock()

            return downloader

    def test_record_rate_limit_event_sets_timestamp(self, mock_downloader_with_checkpoint):
        """_record_rate_limit_event should set timestamp on checkpoint."""
        downloader = mock_downloader_with_checkpoint
        downloader._rate_limit_event_count = 1

        downloader._record_rate_limit_event()

        assert downloader.checkpoint.last_rate_limit_timestamp is not None
        # Verify it's a valid ISO timestamp
        datetime.fromisoformat(downloader.checkpoint.last_rate_limit_timestamp)

    def test_record_rate_limit_event_updates_count(self, mock_downloader_with_checkpoint):
        """_record_rate_limit_event should update event count on checkpoint."""
        downloader = mock_downloader_with_checkpoint
        downloader._rate_limit_event_count = 5

        downloader._record_rate_limit_event()

        assert downloader.checkpoint.rate_limit_event_count == 5

    def test_record_rate_limit_event_saves_checkpoint(self, mock_downloader_with_checkpoint):
        """_record_rate_limit_event should call _save_checkpoint."""
        downloader = mock_downloader_with_checkpoint
        downloader._rate_limit_event_count = 1

        downloader._record_rate_limit_event()

        downloader._save_checkpoint.assert_called_once()

    def test_record_rate_limit_event_no_checkpoint(self):
        """_record_rate_limit_event should handle missing checkpoint gracefully."""
        with patch('src.downloader.core.VideoDownloader.__init__', return_value=None):
            from src.downloader.core import VideoDownloader
            downloader = VideoDownloader.__new__(VideoDownloader)
            downloader._rate_limit_event_count = 1
            downloader.checkpoint = None

            # Should not raise
            downloader._record_rate_limit_event()


# =============================================================================
# Test recovery mode behavior
# =============================================================================

class TestRecoveryModeBehavior:
    """Tests for recovery mode when within cooldown period."""

    @pytest.fixture
    def mock_downloader_for_backoff(self):
        """Create a mock VideoDownloader for testing backoff behavior."""
        with patch('src.downloader.core.VideoDownloader.__init__', return_value=None):
            from src.downloader.core import VideoDownloader, TierRateLimitState
            downloader = VideoDownloader.__new__(VideoDownloader)

            # Set up minimal required attributes
            downloader._rate_limit_backoff_count = 0
            downloader._rate_limit_total_delay = 0.0
            downloader._rate_limit_event_count = 0
            downloader._in_cooldown_recovery_mode = False

            # Per-tier rate limit state (US-001)
            downloader._per_tier_isolation = True
            downloader._tier_rate_limit_states = {
                'short': TierRateLimitState(),
                'medium': TierRateLimitState(),
                'long': TierRateLimitState(),
                'longer': TierRateLimitState(),
            }

            # Create checkpoint
            downloader.checkpoint = DownloadCheckpoint(
                completed_keywords=[],
                completed_videos=[],
                failed_keywords=[],
                current_keyword=None,
                current_tier=None,
                timestamp=datetime.now().isoformat(),
                last_rate_limit_timestamp=None,
                rate_limit_event_count=0
            )

            # Mock save checkpoint
            downloader._save_checkpoint = MagicMock()

            # Mock rate limit config
            rate_limit_config = MagicMock()
            rate_limit_config.initial_backoff_seconds = 5.0
            rate_limit_config.max_backoff_before_rotate = 60.0
            rate_limit_config.backoff_multiplier = 2.0
            rate_limit_config.per_tier_isolation = True

            download_config = MagicMock()
            download_config.rate_limit = rate_limit_config
            downloader.download_config = download_config

            # No cookie rotator or VPN
            downloader.cookie_rotator = None
            downloader.vpn_manager = None

            # Rate limit metrics (US-010)
            from src.downloader.rate_limit_metrics import RateLimitMetrics
            downloader.rate_limit_metrics = RateLimitMetrics()

            return downloader

    @patch('time.sleep')
    def test_recovery_mode_doubles_initial_backoff(self, mock_sleep, mock_downloader_for_backoff):
        """Recovery mode should double initial backoff."""
        downloader = mock_downloader_for_backoff
        downloader._in_cooldown_recovery_mode = True

        # First rate limit error
        downloader.handle_rate_limit_error("429 too many requests")

        # In recovery mode: initial_backoff * 2 = 10s
        mock_sleep.assert_called_once()
        actual_delay = mock_sleep.call_args[0][0]
        assert actual_delay == 10.0  # 5.0 * 2 = 10.0

    @patch('time.sleep')
    def test_normal_mode_uses_standard_backoff(self, mock_sleep, mock_downloader_for_backoff):
        """Normal mode (not recovery) should use standard backoff."""
        downloader = mock_downloader_for_backoff
        downloader._in_cooldown_recovery_mode = False

        # First rate limit error
        downloader.handle_rate_limit_error("429 too many requests")

        # Normal mode: initial_backoff = 5s
        mock_sleep.assert_called_once()
        actual_delay = mock_sleep.call_args[0][0]
        assert actual_delay == 5.0

    @patch('time.sleep')
    def test_recovery_mode_halves_max_backoff(self, mock_sleep, mock_downloader_for_backoff):
        """Recovery mode should halve max_backoff_before_rotate."""
        downloader = mock_downloader_for_backoff
        downloader._in_cooldown_recovery_mode = True

        # Set total delay just under normal max (60) but over recovery max (30)
        downloader._rate_limit_total_delay = 25.0

        # Next backoff would be: 10 * (2^0) = 10, but remaining is 30-25=5
        downloader.handle_rate_limit_error("429 too many requests")

        # Should cap at 5s (remaining under recovery max)
        mock_sleep.assert_called_once()
        actual_delay = mock_sleep.call_args[0][0]
        assert actual_delay == 5.0  # Capped at remaining (30 - 25 = 5)


# =============================================================================
# Test logging
# =============================================================================

class TestCooldownLogging:
    """Tests for cooldown status logging."""

    @pytest.fixture
    def mock_downloader(self):
        """Create a mock VideoDownloader."""
        with patch('src.downloader.core.VideoDownloader.__init__', return_value=None):
            from src.downloader.core import VideoDownloader
            downloader = VideoDownloader.__new__(VideoDownloader)

            # Set up minimal required attributes
            downloader._rate_limit_backoff_count = 0
            downloader._rate_limit_total_delay = 0.0
            downloader._rate_limit_event_count = 0
            downloader._in_cooldown_recovery_mode = False

            # Mock rate limit config
            rate_limit_config = MagicMock()
            rate_limit_config.resume_cooldown_minutes = 15.0

            download_config = MagicMock()
            download_config.rate_limit = rate_limit_config
            downloader.download_config = download_config

            return downloader

    def test_logs_within_cooldown(self, mock_downloader, caplog):
        """Should log warning when within cooldown period."""
        import logging
        caplog.set_level(logging.INFO)

        five_minutes_ago = datetime.now() - timedelta(minutes=5)
        checkpoint = DownloadCheckpoint(
            completed_keywords=[],
            completed_videos=[],
            failed_keywords=[],
            current_keyword=None,
            current_tier=None,
            timestamp=datetime.now().isoformat(),
            last_rate_limit_timestamp=five_minutes_ago.isoformat(),
            rate_limit_event_count=3
        )

        mock_downloader._check_rate_limit_cooldown(checkpoint)

        # Should log about cooldown being active
        assert "Rate limit cooldown" in caplog.text
        assert "recovery mode" in caplog.text.lower()

    def test_logs_cooldown_cleared(self, mock_downloader, caplog):
        """Should log when cooldown period has passed."""
        import logging
        caplog.set_level(logging.INFO)

        twenty_minutes_ago = datetime.now() - timedelta(minutes=20)
        checkpoint = DownloadCheckpoint(
            completed_keywords=[],
            completed_videos=[],
            failed_keywords=[],
            current_keyword=None,
            current_tier=None,
            timestamp=datetime.now().isoformat(),
            last_rate_limit_timestamp=twenty_minutes_ago.isoformat(),
            rate_limit_event_count=5
        )

        mock_downloader._check_rate_limit_cooldown(checkpoint)

        # Should log that cooldown is cleared
        assert "cooldown cleared" in caplog.text.lower()


# =============================================================================
# Test checkpoint roundtrip
# =============================================================================

class TestCheckpointRoundtrip:
    """Tests for checkpoint save/restore with rate limit data."""

    def test_full_checkpoint_roundtrip(self):
        """Checkpoint should preserve all rate limit data through save/restore."""
        timestamp = datetime.now().isoformat()
        rate_limit_ts = (datetime.now() - timedelta(minutes=5)).isoformat()

        original = DownloadCheckpoint(
            completed_keywords=["kw1", "kw2"],
            completed_videos=["v1", "v2"],
            failed_keywords=["f1"],
            current_keyword="current",
            current_tier="medium",
            timestamp=timestamp,
            speed_tracker_state={"samples": 5},
            last_rate_limit_timestamp=rate_limit_ts,
            rate_limit_event_count=7
        )

        # Serialize and deserialize
        data = original.to_dict()
        restored = DownloadCheckpoint.from_dict(data)

        # Verify all fields
        assert restored.completed_keywords == original.completed_keywords
        assert restored.completed_videos == original.completed_videos
        assert restored.failed_keywords == original.failed_keywords
        assert restored.current_keyword == original.current_keyword
        assert restored.current_tier == original.current_tier
        assert restored.timestamp == original.timestamp
        assert restored.speed_tracker_state == original.speed_tracker_state
        assert restored.last_rate_limit_timestamp == original.last_rate_limit_timestamp
        assert restored.rate_limit_event_count == original.rate_limit_event_count


# =============================================================================
# Integration test
# =============================================================================

class TestCooldownIntegration:
    """Integration tests for cooldown tracking flow."""

    def test_rate_limit_increments_event_count(self):
        """handle_rate_limit_error should increment event count."""
        with patch('src.downloader.core.VideoDownloader.__init__', return_value=None):
            from src.downloader.core import VideoDownloader, TierRateLimitState
            downloader = VideoDownloader.__new__(VideoDownloader)

            # Set up minimal required attributes
            downloader._rate_limit_backoff_count = 0
            downloader._rate_limit_total_delay = 0.0
            downloader._rate_limit_event_count = 0
            downloader._in_cooldown_recovery_mode = False

            # Per-tier rate limit state (US-001)
            downloader._per_tier_isolation = True
            downloader._tier_rate_limit_states = {
                'short': TierRateLimitState(),
                'medium': TierRateLimitState(),
                'long': TierRateLimitState(),
                'longer': TierRateLimitState(),
            }

            # Create checkpoint
            downloader.checkpoint = DownloadCheckpoint(
                completed_keywords=[],
                completed_videos=[],
                failed_keywords=[],
                current_keyword=None,
                current_tier=None,
                timestamp=datetime.now().isoformat(),
                last_rate_limit_timestamp=None,
                rate_limit_event_count=0
            )

            # Mock dependencies
            downloader._save_checkpoint = MagicMock()

            rate_limit_config = MagicMock()
            rate_limit_config.initial_backoff_seconds = 5.0
            rate_limit_config.max_backoff_before_rotate = 60.0
            rate_limit_config.backoff_multiplier = 2.0
            rate_limit_config.per_tier_isolation = True

            download_config = MagicMock()
            download_config.rate_limit = rate_limit_config
            downloader.download_config = download_config

            downloader.cookie_rotator = None
            downloader.vpn_manager = None

            # Rate limit metrics (US-010)
            from src.downloader.rate_limit_metrics import RateLimitMetrics
            downloader.rate_limit_metrics = RateLimitMetrics()

            # Simulate multiple rate limit errors
            with patch('time.sleep'):
                downloader.handle_rate_limit_error("429")
                downloader.handle_rate_limit_error("429")
                downloader.handle_rate_limit_error("429")

            # Should have incremented event count
            assert downloader._rate_limit_event_count == 3
            assert downloader.checkpoint.rate_limit_event_count == 3
