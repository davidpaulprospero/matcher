"""
Tests for US-59-008: Deduplicate list-subs calls for same-channel videos in batch fetch.

Verifies that fetch_captions_batch() caches list-subs results per channel,
avoiding redundant yt-dlp calls when multiple videos share the same channel.
"""

import pytest
import threading
from unittest.mock import Mock, patch, MagicMock, call
from typing import List, Optional

from src.caption_fetcher import (
    CaptionFetcher,
    CaptionResult,
    CaptionSegment,
    CaptionUnavailableError,
    CaptionFetchError,
    AvailableLanguage,
)


def _make_caption_result(video_id="vid001", language="en", is_auto=False):
    """Helper to create a CaptionResult with minimal segments."""
    return CaptionResult(
        video_id=video_id,
        segments=[CaptionSegment(index=0, start_time=0.0, end_time=1.0, text="Hello world")],
        language=language,
        is_auto_generated=is_auto,
        format_source="vtt",
    )


def _make_config():
    """Helper to create a mock config for CaptionFetcher."""
    config = Mock()
    caption_first = Mock()
    caption_first.timeout = 30
    caption_first.max_retries = 1
    caption_first.retry_delay = 0.01
    caption_first.preferred_formats = ["vtt"]
    caption_first.adaptive_format_order = False
    caption_first.pre_check_availability = False
    caption_first.format_timeouts = {}
    caption_first.max_parallel_fetches = 1  # Sequential for predictable testing
    caption_first.abort_on_error_pattern = 'skip'
    caption_first.prioritize_by_channel = False
    caption_first.stuck_worker_threshold = 60.0
    caption_first.allow_auto_generated = True
    config.download.caption_first = caption_first
    return config


ENGLISH_AVAILABLE = [
    AvailableLanguage(code='en', name='English', is_auto_generated=False),
]

NO_CAPTIONS = []


@pytest.mark.fast
class TestChannelListSubsDedup:
    """Tests for channel-based list-subs deduplication (US-59-008)."""

    def test_same_channel_dedup_with_captions(self):
        """5 videos from the same channel result in only 1 list-subs call when captions exist.

        AC: channel_caption_availability dict caches the list-subs result per channel_id.
        """
        config = _make_config()
        fetcher = CaptionFetcher(config=config)

        video_ids = ['vid001', 'vid002', 'vid003', 'vid004', 'vid005']
        channel_map = {vid: 'UC_TestChannel' for vid in video_ids}
        fetcher.set_video_channel_map(channel_map)

        list_subs_call_count = 0

        def mock_list_languages(video_id):
            nonlocal list_subs_call_count
            list_subs_call_count += 1
            return list(ENGLISH_AVAILABLE)

        def mock_fetch_captions(video_id, language='en', prefer_manual=True):
            return _make_caption_result(video_id=video_id)

        with patch.object(fetcher, 'list_available_languages', side_effect=mock_list_languages), \
             patch.object(fetcher, 'fetch_captions', side_effect=mock_fetch_captions), \
             patch.object(fetcher, 'select_best_language', return_value=ENGLISH_AVAILABLE[0]):

            results = fetcher.fetch_captions_batch(
                video_ids=video_ids,
                preferred_language='en',
                max_workers=1,
            )

        # All 5 should succeed
        assert len(results) == 5
        for vid in video_ids:
            assert isinstance(results[vid], CaptionResult)

        # Only 1 list-subs call (first video populates cache, rest use cache)
        assert list_subs_call_count == 1

    def test_same_channel_no_captions_max_two_calls(self):
        """5 videos from same channel with no captions result in at most 2 list-subs calls.

        AC: Unit test verifies that 5 videos from the same channel result in at most
        2 list-subs calls (first video + one retry after unavailable).
        """
        config = _make_config()
        fetcher = CaptionFetcher(config=config)

        video_ids = ['vid001', 'vid002', 'vid003', 'vid004', 'vid005']
        channel_map = {vid: 'UC_NoSubsChannel' for vid in video_ids}
        fetcher.set_video_channel_map(channel_map)

        list_subs_call_count = 0

        def mock_list_languages(video_id):
            nonlocal list_subs_call_count
            list_subs_call_count += 1
            return []  # No captions

        with patch.object(fetcher, 'list_available_languages', side_effect=mock_list_languages):

            results = fetcher.fetch_captions_batch(
                video_ids=video_ids,
                preferred_language='en',
                max_workers=1,
            )

        # All should be unavailable
        assert len(results) == 5
        for vid in video_ids:
            assert not isinstance(results[vid], CaptionResult)
            result = results[vid]
            assert result.get('unavailable', False) or result.get('error', False)

        # At most 2 list-subs calls (1st check + 1 retry)
        assert list_subs_call_count <= 2

    def test_no_channel_map_no_dedup(self):
        """Without channel map, every video makes its own list-subs call.

        AC: The channel cache only used when channel info is available.
        """
        config = _make_config()
        fetcher = CaptionFetcher(config=config)
        # Don't call set_video_channel_map

        video_ids = ['vid001', 'vid002', 'vid003']

        list_subs_call_count = 0

        def mock_list_languages(video_id):
            nonlocal list_subs_call_count
            list_subs_call_count += 1
            return list(ENGLISH_AVAILABLE)

        def mock_fetch_captions(video_id, language='en', prefer_manual=True):
            return _make_caption_result(video_id=video_id)

        with patch.object(fetcher, 'list_available_languages', side_effect=mock_list_languages), \
             patch.object(fetcher, 'fetch_captions', side_effect=mock_fetch_captions), \
             patch.object(fetcher, 'select_best_language', return_value=ENGLISH_AVAILABLE[0]):

            results = fetcher.fetch_captions_batch(
                video_ids=video_ids,
                preferred_language='en',
                max_workers=1,
            )

        # Each video makes its own call (no dedup)
        assert list_subs_call_count == 3

    def test_different_channels_separate_calls(self):
        """Videos from different channels each make their own list-subs call.

        AC: Cache is per-channel, so different channels are checked independently.
        """
        config = _make_config()
        fetcher = CaptionFetcher(config=config)

        video_ids = ['vid001', 'vid002', 'vid003']
        channel_map = {
            'vid001': 'UC_ChannelA',
            'vid002': 'UC_ChannelB',
            'vid003': 'UC_ChannelC',
        }
        fetcher.set_video_channel_map(channel_map)

        list_subs_call_count = 0

        def mock_list_languages(video_id):
            nonlocal list_subs_call_count
            list_subs_call_count += 1
            return list(ENGLISH_AVAILABLE)

        def mock_fetch_captions(video_id, language='en', prefer_manual=True):
            return _make_caption_result(video_id=video_id)

        with patch.object(fetcher, 'list_available_languages', side_effect=mock_list_languages), \
             patch.object(fetcher, 'fetch_captions', side_effect=mock_fetch_captions), \
             patch.object(fetcher, 'select_best_language', return_value=ENGLISH_AVAILABLE[0]):

            results = fetcher.fetch_captions_batch(
                video_ids=video_ids,
                preferred_language='en',
                max_workers=1,
            )

        # 3 different channels = 3 calls
        assert list_subs_call_count == 3
        assert len(results) == 3

    def test_cache_is_per_batch(self):
        """Channel cache is reset between batch calls.

        AC: The channel cache is per-batch (not persisted).
        """
        config = _make_config()
        fetcher = CaptionFetcher(config=config)

        video_ids_batch1 = ['vid001', 'vid002']
        video_ids_batch2 = ['vid003', 'vid004']
        channel_map = {vid: 'UC_SameChannel' for vid in video_ids_batch1 + video_ids_batch2}
        fetcher.set_video_channel_map(channel_map)

        list_subs_call_count = 0

        def mock_list_languages(video_id):
            nonlocal list_subs_call_count
            list_subs_call_count += 1
            return list(ENGLISH_AVAILABLE)

        def mock_fetch_captions(video_id, language='en', prefer_manual=True):
            return _make_caption_result(video_id=video_id)

        with patch.object(fetcher, 'list_available_languages', side_effect=mock_list_languages), \
             patch.object(fetcher, 'fetch_captions', side_effect=mock_fetch_captions), \
             patch.object(fetcher, 'select_best_language', return_value=ENGLISH_AVAILABLE[0]):

            # Batch 1
            fetcher.fetch_captions_batch(
                video_ids=video_ids_batch1,
                preferred_language='en',
                max_workers=1,
            )

            # Batch 2 - cache should be reset, so a new call is needed
            fetcher.fetch_captions_batch(
                video_ids=video_ids_batch2,
                preferred_language='en',
                max_workers=1,
            )

        # 1 call per batch (cache reset between batches)
        assert list_subs_call_count == 2

    def test_channel_cache_allows_override_on_unavailable(self):
        """When channel cache says unavailable, one retry is allowed per channel.

        AC: Individual videos can still override if the channel cache shows
        'unavailable' but video-specific data hasn't been checked.
        """
        config = _make_config()
        fetcher = CaptionFetcher(config=config)

        video_ids = ['vid001', 'vid002', 'vid003']
        channel_map = {vid: 'UC_FlakeyChannel' for vid in video_ids}
        fetcher.set_video_channel_map(channel_map)

        list_subs_call_count = 0

        def mock_list_languages(video_id):
            nonlocal list_subs_call_count
            list_subs_call_count += 1
            # First call returns empty, retry also returns empty
            return []

        with patch.object(fetcher, 'list_available_languages', side_effect=mock_list_languages):

            results = fetcher.fetch_captions_batch(
                video_ids=video_ids,
                preferred_language='en',
                max_workers=1,
            )

        # First video: 1 call (empty → cached)
        # Second video: 1 retry allowed (still empty → now call_count=2)
        # Third video: no call (call_count=2 already, skip)
        assert list_subs_call_count == 2
        assert len(results) == 3

    def test_extract_channel_from_video_search_results(self):
        """fetch_captions_batch extracts channel ID from _video_channel_map.

        AC: fetch_captions_batch() extracts channel ID from video_search_results
        metadata when available.
        """
        config = _make_config()
        fetcher = CaptionFetcher(config=config)

        # Set up channel map (normally done by caption_stage from video_search_results)
        channel_map = {
            'vid001': 'NatGeo',
            'vid002': 'NatGeo',
            'vid003': 'BBC',
        }
        fetcher.set_video_channel_map(channel_map)

        # Verify the map is accessible
        assert fetcher._get_channel_id_from_video('vid001') == 'NatGeo'
        assert fetcher._get_channel_id_from_video('vid002') == 'NatGeo'
        assert fetcher._get_channel_id_from_video('vid003') == 'BBC'
        assert fetcher._get_channel_id_from_video('vid999') is None

    def test_list_languages_with_channel_cache_no_cache(self):
        """Without channel cache active, _list_languages_with_channel_cache calls directly."""
        fetcher = CaptionFetcher()

        # No cache set
        with patch.object(fetcher, 'list_available_languages', return_value=ENGLISH_AVAILABLE) as mock_list:
            result = fetcher._list_languages_with_channel_cache('vid001')
            assert result == ENGLISH_AVAILABLE
            mock_list.assert_called_once_with('vid001')

    def test_list_languages_with_channel_cache_hit(self):
        """With channel cache, cached result is returned without calling list_available_languages."""
        fetcher = CaptionFetcher()
        fetcher._channel_caption_cache = {
            'UC_TestChannel': {
                'languages': list(ENGLISH_AVAILABLE),
                'call_count': 1,
            }
        }
        fetcher._channel_cache_lock = threading.Lock()
        fetcher._video_channel_map = {'vid001': 'UC_TestChannel'}

        with patch.object(fetcher, 'list_available_languages') as mock_list:
            result = fetcher._list_languages_with_channel_cache('vid001')
            assert len(result) == 1
            assert result[0].code == 'en'
            # Should NOT have called list_available_languages
            mock_list.assert_not_called()

    def test_list_languages_with_channel_cache_unavailable_retry(self):
        """Channel cache with empty result allows one retry before skipping."""
        fetcher = CaptionFetcher()
        fetcher._channel_caption_cache = {
            'UC_EmptyChannel': {
                'languages': [],
                'call_count': 1,  # First call was made
            }
        }
        fetcher._channel_cache_lock = threading.Lock()
        fetcher._video_channel_map = {'vid002': 'UC_EmptyChannel'}

        # Retry returns captions this time
        with patch.object(fetcher, 'list_available_languages', return_value=list(ENGLISH_AVAILABLE)) as mock_list:
            result = fetcher._list_languages_with_channel_cache('vid002')
            assert len(result) == 1
            # Should have called list_available_languages (retry)
            mock_list.assert_called_once_with('vid002')

        # Cache should be updated
        assert fetcher._channel_caption_cache['UC_EmptyChannel']['call_count'] == 2
        assert len(fetcher._channel_caption_cache['UC_EmptyChannel']['languages']) == 1

    def test_list_languages_with_channel_cache_unavailable_skip(self):
        """After 2 calls showing empty, subsequent videos are skipped."""
        fetcher = CaptionFetcher()
        fetcher._channel_caption_cache = {
            'UC_EmptyChannel': {
                'languages': [],
                'call_count': 2,  # Already retried
            }
        }
        fetcher._channel_cache_lock = threading.Lock()
        fetcher._video_channel_map = {'vid003': 'UC_EmptyChannel'}

        with patch.object(fetcher, 'list_available_languages') as mock_list:
            with pytest.raises(CaptionUnavailableError):
                fetcher._list_languages_with_channel_cache('vid003')
            # Should NOT have called list_available_languages
            mock_list.assert_not_called()
