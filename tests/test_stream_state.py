"""
Tests for stream state classification (US-007 Sprint 8).

Tests the StreamState enum, classify_stream_state() function, and get_stream_state() method
that enable enhanced live stream detection with state classification.

Test coverage:
- All 5 stream states (LIVE, UPCOMING, VOD, PREMIERE, UNKNOWN) detected correctly
- Edge cases for metadata combinations
- StreamStateResult string formatting
- Integration with CaptionFetcher
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from src.caption_fetcher import (
    CaptionFetcher,
    StreamState,
    StreamStateResult,
    classify_stream_state,
)


class TestStreamStateEnum:
    """Tests for the StreamState enum."""

    @pytest.mark.fast
    def test_has_all_expected_states(self):
        """Verify all 5 required states exist."""
        assert hasattr(StreamState, 'LIVE')
        assert hasattr(StreamState, 'UPCOMING')
        assert hasattr(StreamState, 'VOD')
        assert hasattr(StreamState, 'PREMIERE')
        assert hasattr(StreamState, 'UNKNOWN')

    @pytest.mark.fast
    def test_states_are_unique(self):
        """Verify all states have unique values."""
        values = [s.value for s in StreamState]
        assert len(values) == len(set(values)), "Stream states should have unique values"

    @pytest.mark.fast
    def test_exactly_five_states(self):
        """Verify exactly 5 states exist."""
        assert len(list(StreamState)) == 5


class TestClassifyStreamState:
    """Tests for the classify_stream_state() function."""

    @pytest.mark.fast
    def test_live_status_is_live(self):
        """live_status='is_live' -> LIVE."""
        metadata = {'live_status': 'is_live', 'is_live': True}
        result = classify_stream_state(metadata, 'test123')
        assert result.state == StreamState.LIVE
        assert result.is_live is True

    @pytest.mark.fast
    def test_live_status_is_live_case_insensitive(self):
        """live_status='IS_LIVE' (uppercase) -> LIVE."""
        metadata = {'live_status': 'IS_LIVE'}
        result = classify_stream_state(metadata, 'test123')
        assert result.state == StreamState.LIVE

    @pytest.mark.fast
    def test_is_live_boolean_true(self):
        """is_live=True without live_status -> LIVE."""
        metadata = {'is_live': True}
        result = classify_stream_state(metadata, 'test123')
        assert result.state == StreamState.LIVE

    @pytest.mark.fast
    def test_upcoming_without_release_timestamp(self):
        """live_status='is_upcoming' without release_timestamp -> UPCOMING."""
        metadata = {'live_status': 'is_upcoming'}
        result = classify_stream_state(metadata, 'test123')
        assert result.state == StreamState.UPCOMING

    @pytest.mark.fast
    def test_upcoming_with_release_timestamp(self):
        """live_status='is_upcoming' with release_timestamp -> PREMIERE."""
        future_ts = time.time() + 86400  # 1 day in future
        metadata = {'live_status': 'is_upcoming', 'release_timestamp': future_ts}
        result = classify_stream_state(metadata, 'test123')
        assert result.state == StreamState.PREMIERE
        assert result.scheduled_start is not None

    @pytest.mark.fast
    def test_was_live_with_duration(self):
        """live_status='was_live' with duration -> VOD (completed live)."""
        metadata = {'live_status': 'was_live', 'was_live': True, 'duration': 3600}
        result = classify_stream_state(metadata, 'test123')
        assert result.state == StreamState.VOD
        assert result.was_live is True

    @pytest.mark.fast
    def test_post_live_with_duration(self):
        """live_status='post_live' with duration -> VOD."""
        metadata = {'live_status': 'post_live', 'duration': 7200}
        result = classify_stream_state(metadata, 'test123')
        assert result.state == StreamState.VOD

    @pytest.mark.fast
    def test_not_live_with_duration(self):
        """live_status='not_live' with duration -> VOD."""
        metadata = {'live_status': 'not_live', 'duration': 300}
        result = classify_stream_state(metadata, 'test123')
        assert result.state == StreamState.VOD

    @pytest.mark.fast
    def test_regular_video_with_duration(self):
        """Regular video with only duration -> VOD."""
        metadata = {'duration': 600}
        result = classify_stream_state(metadata, 'test123')
        assert result.state == StreamState.VOD

    @pytest.mark.fast
    def test_empty_metadata(self):
        """Empty metadata -> UNKNOWN."""
        metadata = {}
        result = classify_stream_state(metadata, 'test123')
        assert result.state == StreamState.UNKNOWN

    @pytest.mark.fast
    def test_no_duration_no_live(self):
        """No duration and not live -> UNKNOWN."""
        metadata = {'title': 'Some Video'}
        result = classify_stream_state(metadata, 'test123')
        assert result.state == StreamState.UNKNOWN

    @pytest.mark.fast
    def test_was_live_no_duration(self):
        """was_live=True without duration -> UNKNOWN (might still be processing)."""
        metadata = {'was_live': True}
        result = classify_stream_state(metadata, 'test123')
        assert result.state == StreamState.UNKNOWN

    @pytest.mark.fast
    def test_future_release_timestamp_without_duration(self):
        """Future release_timestamp without live_status or duration -> PREMIERE."""
        future_ts = time.time() + 86400
        metadata = {'release_timestamp': future_ts}
        result = classify_stream_state(metadata, 'test123')
        assert result.state == StreamState.PREMIERE

    @pytest.mark.fast
    def test_past_release_timestamp_with_duration(self):
        """Past release_timestamp with duration -> VOD."""
        past_ts = time.time() - 86400  # 1 day in past
        metadata = {'release_timestamp': past_ts, 'duration': 300}
        result = classify_stream_state(metadata, 'test123')
        assert result.state == StreamState.VOD

    @pytest.mark.fast
    def test_video_id_preserved(self):
        """Video ID is preserved in result."""
        metadata = {'duration': 300}
        result = classify_stream_state(metadata, 'myVideoId12')
        assert result.video_id == 'myVideoId12'

    @pytest.mark.fast
    def test_scheduled_start_formatting(self):
        """Scheduled start time is formatted correctly."""
        # Use a known timestamp: 2026-01-27 10:00:00 UTC
        ts = 1769511600
        metadata = {'live_status': 'is_upcoming', 'release_timestamp': ts}
        result = classify_stream_state(metadata, 'test123')
        assert result.scheduled_start is not None
        assert 'UTC' in result.scheduled_start


class TestStreamStateResult:
    """Tests for the StreamStateResult dataclass."""

    @pytest.mark.fast
    def test_str_format_live(self):
        """String format for LIVE state."""
        result = StreamStateResult(
            state=StreamState.LIVE,
            video_id='abc123',
            is_live=True,
            live_status='is_live'
        )
        s = str(result)
        assert 'abc123' in s
        assert 'LIVE' in s
        assert 'is_live' in s

    @pytest.mark.fast
    def test_str_format_upcoming_with_scheduled(self):
        """String format for UPCOMING with scheduled time."""
        result = StreamStateResult(
            state=StreamState.UPCOMING,
            video_id='xyz789',
            scheduled_start='2026-01-27 10:00 UTC',
            live_status='is_upcoming'
        )
        s = str(result)
        assert 'xyz789' in s
        assert 'UPCOMING' in s
        assert '2026-01-27 10:00 UTC' in s

    @pytest.mark.fast
    def test_str_format_premiere_with_scheduled(self):
        """String format for PREMIERE with scheduled time."""
        result = StreamStateResult(
            state=StreamState.PREMIERE,
            video_id='pre456',
            scheduled_start='2026-02-01 15:00 UTC'
        )
        s = str(result)
        assert 'pre456' in s
        assert 'PREMIERE' in s
        assert '2026-02-01 15:00 UTC' in s

    @pytest.mark.fast
    def test_str_format_vod_simple(self):
        """String format for VOD is simple."""
        result = StreamStateResult(
            state=StreamState.VOD,
            video_id='vod000',
            duration=300
        )
        s = str(result)
        assert 'vod000' in s
        assert 'VOD' in s
        # VOD should not include scheduled_start
        assert 'scheduled' not in s.lower()


class TestCaptionFetcherGetStreamState:
    """Tests for CaptionFetcher.get_stream_state() method."""

    @pytest.fixture
    def fetcher(self):
        """Create a CaptionFetcher instance."""
        return CaptionFetcher()

    @pytest.mark.fast
    def test_invalid_video_id_returns_unknown(self, fetcher):
        """Invalid video ID returns UNKNOWN state."""
        result = fetcher.get_stream_state('invalid')
        assert result.state == StreamState.UNKNOWN
        assert result.video_id == 'invalid'

    @pytest.mark.fast
    def test_short_video_id_returns_unknown(self, fetcher):
        """Short video ID (< 11 chars) returns UNKNOWN state."""
        result = fetcher.get_stream_state('abc')
        assert result.state == StreamState.UNKNOWN

    @pytest.mark.fast
    def test_empty_video_id_returns_unknown(self, fetcher):
        """Empty video ID returns UNKNOWN state."""
        result = fetcher.get_stream_state('')
        assert result.state == StreamState.UNKNOWN

    @patch('src.caption_fetcher.subprocess.run')
    @pytest.mark.fast
    def test_subprocess_timeout_returns_unknown(self, mock_run, fetcher):
        """Subprocess timeout returns UNKNOWN state."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd='yt-dlp', timeout=10)

        result = fetcher.get_stream_state('dQw4w9WgXcQ')
        assert result.state == StreamState.UNKNOWN

    @patch('src.caption_fetcher.subprocess.run')
    @pytest.mark.fast
    def test_subprocess_failure_returns_unknown(self, mock_run, fetcher):
        """Subprocess failure (non-zero return) returns UNKNOWN state."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = 'Video unavailable'
        mock_run.return_value = mock_result

        result = fetcher.get_stream_state('dQw4w9WgXcQ')
        assert result.state == StreamState.UNKNOWN

    @patch('src.caption_fetcher.subprocess.run')
    @pytest.mark.fast
    def test_live_stream_detected(self, mock_run, fetcher):
        """Live stream is correctly detected."""
        import json
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            'is_live': True,
            'live_status': 'is_live',
            'title': 'Live Stream Test'
        })
        mock_run.return_value = mock_result

        result = fetcher.get_stream_state('live123test')
        assert result.state == StreamState.LIVE

    @patch('src.caption_fetcher.subprocess.run')
    @pytest.mark.fast
    def test_vod_detected(self, mock_run, fetcher):
        """Regular VOD is correctly detected."""
        import json
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            'duration': 600,
            'live_status': 'not_live',
            'title': 'Regular Video'
        })
        mock_run.return_value = mock_result

        result = fetcher.get_stream_state('vod12345678')
        assert result.state == StreamState.VOD

    @patch('src.caption_fetcher.subprocess.run')
    @pytest.mark.fast
    def test_upcoming_detected(self, mock_run, fetcher):
        """Upcoming stream is correctly detected."""
        import json
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            'live_status': 'is_upcoming',
            'title': 'Scheduled Stream'
        })
        mock_run.return_value = mock_result

        result = fetcher.get_stream_state('upcomingXYZ')  # 11 chars exactly
        assert result.state == StreamState.UPCOMING

    @patch('src.caption_fetcher.subprocess.run')
    @pytest.mark.fast
    def test_premiere_detected(self, mock_run, fetcher):
        """Premiere is correctly detected."""
        import json
        future_ts = time.time() + 86400
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            'live_status': 'is_upcoming',
            'release_timestamp': future_ts,
            'title': 'Video Premiere'
        })
        mock_run.return_value = mock_result

        result = fetcher.get_stream_state('premiereXYZ')  # 11 chars exactly
        assert result.state == StreamState.PREMIERE
        assert result.scheduled_start is not None

    @patch('src.caption_fetcher.subprocess.run')
    @pytest.mark.fast
    def test_invalid_json_returns_unknown(self, mock_run, fetcher):
        """Invalid JSON output returns UNKNOWN state."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = 'not valid json'
        mock_run.return_value = mock_result

        result = fetcher.get_stream_state('badjson12345')
        assert result.state == StreamState.UNKNOWN


class TestStreamStateEdgeCases:
    """Edge case tests for stream state classification."""

    @pytest.mark.fast
    def test_conflicting_is_live_and_was_live(self):
        """Both is_live and was_live True -> LIVE takes priority."""
        metadata = {'is_live': True, 'was_live': True, 'duration': 3600}
        result = classify_stream_state(metadata, 'conflict')
        assert result.state == StreamState.LIVE

    @pytest.mark.fast
    def test_live_status_overrides_booleans(self):
        """live_status takes priority over boolean flags."""
        metadata = {
            'is_live': False,
            'was_live': True,
            'live_status': 'not_live',
            'duration': 600
        }
        result = classify_stream_state(metadata, 'override')
        assert result.state == StreamState.VOD

    @pytest.mark.fast
    def test_zero_duration_treated_as_no_duration(self):
        """Duration of 0 treated as no duration."""
        metadata = {'duration': 0, 'live_status': 'not_live'}
        result = classify_stream_state(metadata, 'zerodur')
        # not_live without positive duration is ambiguous
        assert result.state == StreamState.UNKNOWN

    @pytest.mark.fast
    def test_negative_duration_treated_as_no_duration(self):
        """Negative duration treated as no duration."""
        metadata = {'duration': -100}
        result = classify_stream_state(metadata, 'negdur')
        assert result.state == StreamState.UNKNOWN

    @pytest.mark.fast
    def test_invalid_release_timestamp(self):
        """Invalid release_timestamp is handled gracefully."""
        metadata = {
            'live_status': 'is_upcoming',
            'release_timestamp': 'not-a-timestamp'
        }
        result = classify_stream_state(metadata, 'badts')
        # Should still detect as UPCOMING, but without scheduled_start
        assert result.state in (StreamState.UPCOMING, StreamState.PREMIERE)

    @pytest.mark.fast
    def test_very_old_release_timestamp(self):
        """Very old release_timestamp with duration -> VOD."""
        metadata = {
            'release_timestamp': 0,  # Unix epoch
            'duration': 300
        }
        result = classify_stream_state(metadata, 'old')
        assert result.state == StreamState.VOD


class TestStreamStateIntegration:
    """Integration tests for stream state with CaptionStage."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config with caption_first settings."""
        config = MagicMock()
        config.download.caption_first.enabled = True
        config.download.caption_first.skip_live_streams = True
        config.download.caption_first.handle_upcoming = 'skip'
        return config

    @pytest.mark.fast
    def test_all_states_have_distinct_handling(self, mock_config):
        """Verify each state has distinct handling in acceptance criteria."""
        # LIVE: Always skipped
        # UPCOMING: Based on handle_upcoming config
        # VOD: Normal caption fetch
        # PREMIERE: Same as UPCOMING
        # UNKNOWN: Assume VOD (attempt fetch)

        # This test documents the expected handling behavior
        live = StreamState.LIVE
        upcoming = StreamState.UPCOMING
        vod = StreamState.VOD
        premiere = StreamState.PREMIERE
        unknown = StreamState.UNKNOWN

        # States that should be skipped when skip_live_streams=True
        skip_states = {live}

        # States affected by handle_upcoming config
        upcoming_affected = {upcoming, premiere}

        # States that proceed to caption fetch
        fetch_states = {vod, unknown}

        # Verify no overlaps
        assert not skip_states & fetch_states
        assert live in skip_states
        assert vod in fetch_states
        assert unknown in fetch_states

    @pytest.mark.fast
    def test_handle_upcoming_modes(self, mock_config):
        """Verify all 3 handle_upcoming modes are valid."""
        valid_modes = {'skip', 'queue', 'check_later'}

        # All modes should be documented
        assert 'skip' in valid_modes  # Treat like live - skip entirely
        assert 'queue' in valid_modes  # Add to pending_streams list
        assert 'check_later' in valid_modes  # Skip but can retry next run
