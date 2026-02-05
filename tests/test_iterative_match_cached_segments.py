"""
Tests for iterative match handling of cached caption segments.

Verifies that _fetch_captions_for_videos correctly handles both:
- CaptionSegment objects (from fresh fetches via CaptionResult)
- Dict segments (from cache via CachedCaption)

Bug: 'dict' object has no attribute 'end_time' when cache returns dicts
Fix: isinstance check to branch between attribute and dict key access
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock, patch, PropertyMock
import pytest

from src.stages.iterative_match import IterativeMatchStage


# ============================================================================
# Fixtures
# ============================================================================

@dataclass
class FakeCaptionSegment:
    """Mimics CaptionSegment with attribute access."""
    index: int
    start_time: float
    end_time: float
    text: str
    source_file: str = ""


@dataclass
class FakeCaptionResult:
    """Mimics CaptionResult with CaptionSegment objects."""
    video_id: str
    segments: List[FakeCaptionSegment]
    language: str = "en"
    is_auto_generated: bool = True
    format_source: str = "srv3"


@dataclass
class FakeCachedCaption:
    """Mimics CachedCaption with dict segments."""
    video_id: str
    segments: List[Dict[str, Any]]
    language: str = "en"
    is_auto_generated: bool = True
    format_source: str = "srv3"
    fetch_timestamp: float = 1000000.0
    duration: float = 60.0


def make_object_segments(count=3):
    """Create CaptionSegment-like objects."""
    return [
        FakeCaptionSegment(
            index=i,
            start_time=i * 10.0,
            end_time=(i + 1) * 10.0,
            text=f"Segment {i}"
        )
        for i in range(count)
    ]


def make_dict_segments(count=3):
    """Create dict segments as stored in cache (CaptionSegment.to_dict() format)."""
    return [
        {
            'index': i,
            'start': i * 10.0,
            'end': (i + 1) * 10.0,
            'text': f"Segment {i}",
            'source_file': '',
        }
        for i in range(count)
    ]


@pytest.fixture
def stage():
    """Create IterativeMatchStage with cookie rotator initialized."""
    s = IterativeMatchStage()
    s._cookie_rotator = MagicMock()
    s._cookie_rotator.get_cookie_args.return_value = []
    return s


@pytest.fixture
def mock_config():
    """Config with caption_first settings."""
    config = MagicMock()
    config.download.caption_first = MagicMock()
    config.download.caption_first.preferred_language = 'en'
    return config


@pytest.fixture
def mock_iter_config():
    """Iterative matching config."""
    ic = MagicMock()
    ic.caption_batch_size = 10
    ic.caption_fetch_delay = 0.0
    return ic


# ============================================================================
# Tests: Dict segments from cache
# ============================================================================

class TestCachedDictSegments:
    """Verify _fetch_captions_for_videos handles dict segments from cache."""

    @patch('src.caption_fetcher.CaptionFetcher')
    @patch('src.caption_fetcher.CaptionCache')
    @patch('src.caption_fetcher.determine_caption_quality', return_value='medium')
    def test_dict_segments_from_cache_no_attribute_error(
        self, mock_quality_fn, MockCache, MockFetcher, stage, mock_config, mock_iter_config
    ):
        """Dict segments from cache must not raise 'dict has no attribute end_time'."""
        cache_instance = MockCache.return_value
        cache_instance.enabled = True
        cache_instance.get_caption.return_value = FakeCachedCaption(
            video_id='test123',
            segments=make_dict_segments(3),
        )

        fetcher_instance = MockFetcher.return_value

        result = stage._fetch_captions_for_videos(
            video_ids=['test123'],
            config=mock_config,
            iter_config=mock_iter_config,
        )

        assert len(result) == 1
        assert result[0]['video_id'] == 'test123'
        # Fetcher should NOT have been called (cache hit)
        fetcher_instance.fetch_captions.assert_not_called()

    @patch('src.caption_fetcher.CaptionFetcher')
    @patch('src.caption_fetcher.CaptionCache')
    @patch('src.caption_fetcher.determine_caption_quality', return_value='medium')
    def test_dict_segments_produce_correct_output_format(
        self, mock_quality_fn, MockCache, MockFetcher, stage, mock_config, mock_iter_config
    ):
        """Dict segments must be normalized to {text, start, end} output format."""
        cache_instance = MockCache.return_value
        cache_instance.enabled = True
        cache_instance.get_caption.return_value = FakeCachedCaption(
            video_id='vid1',
            segments=make_dict_segments(2),
        )

        result = stage._fetch_captions_for_videos(
            video_ids=['vid1'],
            config=mock_config,
            iter_config=mock_iter_config,
        )

        segments = result[0]['segments']
        assert len(segments) == 2
        assert segments[0] == {'text': 'Segment 0', 'start': 0.0, 'end': 10.0}
        assert segments[1] == {'text': 'Segment 1', 'start': 10.0, 'end': 20.0}

    @patch('src.caption_fetcher.CaptionFetcher')
    @patch('src.caption_fetcher.CaptionCache')
    @patch('src.caption_fetcher.determine_caption_quality', return_value='high')
    def test_dict_segments_duration_calculated_correctly(
        self, mock_quality_fn, MockCache, MockFetcher, stage, mock_config, mock_iter_config
    ):
        """Duration sum from dict segments must use 'end' - 'start' keys."""
        dict_segs = [
            {'index': 0, 'start': 0.0, 'end': 5.0, 'text': 'A', 'source_file': ''},
            {'index': 1, 'start': 5.0, 'end': 15.0, 'text': 'B', 'source_file': ''},
            {'index': 2, 'start': 20.0, 'end': 30.0, 'text': 'C', 'source_file': ''},
        ]
        cache_instance = MockCache.return_value
        cache_instance.enabled = True
        cache_instance.get_caption.return_value = FakeCachedCaption(
            video_id='vid2',
            segments=dict_segs,
        )

        stage._fetch_captions_for_videos(
            video_ids=['vid2'],
            config=mock_config,
            iter_config=mock_iter_config,
        )

        # total_duration should be (5-0) + (15-5) + (30-20) = 25.0
        mock_quality_fn.assert_called_once_with(
            is_auto_generated=True,
            segment_count=3,
            total_duration=25.0,
        )


# ============================================================================
# Tests: Object segments from fresh fetch
# ============================================================================

class TestFreshObjectSegments:
    """Verify _fetch_captions_for_videos still works with CaptionSegment objects."""

    @patch('src.caption_fetcher.CaptionFetcher')
    @patch('src.caption_fetcher.CaptionCache')
    @patch('src.caption_fetcher.determine_caption_quality', return_value='medium')
    def test_object_segments_from_fetcher_work(
        self, mock_quality_fn, MockCache, MockFetcher, stage, mock_config, mock_iter_config
    ):
        """CaptionSegment objects from fresh fetch must work as before."""
        cache_instance = MockCache.return_value
        cache_instance.enabled = True
        cache_instance.get_caption.return_value = None  # Cache miss

        fetcher_instance = MockFetcher.return_value
        fetcher_instance.fetch_captions.return_value = FakeCaptionResult(
            video_id='fresh1',
            segments=make_object_segments(2),
        )

        result = stage._fetch_captions_for_videos(
            video_ids=['fresh1'],
            config=mock_config,
            iter_config=mock_iter_config,
        )

        assert len(result) == 1
        assert result[0]['video_id'] == 'fresh1'
        segments = result[0]['segments']
        assert segments[0] == {'text': 'Segment 0', 'start': 0.0, 'end': 10.0}
        assert segments[1] == {'text': 'Segment 1', 'start': 10.0, 'end': 20.0}

    @patch('src.caption_fetcher.CaptionFetcher')
    @patch('src.caption_fetcher.CaptionCache')
    @patch('src.caption_fetcher.determine_caption_quality', return_value='high')
    def test_object_segments_duration_calculated_correctly(
        self, mock_quality_fn, MockCache, MockFetcher, stage, mock_config, mock_iter_config
    ):
        """Duration sum from object segments must use .end_time - .start_time."""
        obj_segs = [
            FakeCaptionSegment(index=0, start_time=0.0, end_time=8.0, text='X'),
            FakeCaptionSegment(index=1, start_time=10.0, end_time=22.0, text='Y'),
        ]
        cache_instance = MockCache.return_value
        cache_instance.enabled = True
        cache_instance.get_caption.return_value = None

        fetcher_instance = MockFetcher.return_value
        fetcher_instance.fetch_captions.return_value = FakeCaptionResult(
            video_id='fresh2',
            segments=obj_segs,
        )

        stage._fetch_captions_for_videos(
            video_ids=['fresh2'],
            config=mock_config,
            iter_config=mock_iter_config,
        )

        # total_duration = (8-0) + (22-10) = 20.0
        mock_quality_fn.assert_called_once_with(
            is_auto_generated=True,
            segment_count=2,
            total_duration=20.0,
        )


# ============================================================================
# Tests: Mixed scenarios
# ============================================================================

class TestMixedCacheAndFresh:
    """Verify handling when some videos are cached (dicts) and some are fresh (objects)."""

    @patch('src.caption_fetcher.CaptionFetcher')
    @patch('src.caption_fetcher.CaptionCache')
    @patch('src.caption_fetcher.determine_caption_quality', return_value='medium')
    def test_mixed_cache_and_fresh_both_succeed(
        self, mock_quality_fn, MockCache, MockFetcher, stage, mock_config, mock_iter_config
    ):
        """Mix of cached dicts and fresh objects must all produce valid output."""
        cache_instance = MockCache.return_value
        cache_instance.enabled = True

        # First video: cache hit (dicts), second: cache miss (objects)
        cache_instance.get_caption.side_effect = [
            FakeCachedCaption(video_id='cached1', segments=make_dict_segments(2)),
            None,  # Cache miss for fresh1
        ]

        fetcher_instance = MockFetcher.return_value
        fetcher_instance.fetch_captions.return_value = FakeCaptionResult(
            video_id='fresh1',
            segments=make_object_segments(2),
        )

        result = stage._fetch_captions_for_videos(
            video_ids=['cached1', 'fresh1'],
            config=mock_config,
            iter_config=mock_iter_config,
        )

        assert len(result) == 2
        # Both should have normalized segment format
        for candidate in result:
            for seg in candidate['segments']:
                assert 'text' in seg
                assert 'start' in seg
                assert 'end' in seg
                assert isinstance(seg['start'], float)
                assert isinstance(seg['end'], float)
