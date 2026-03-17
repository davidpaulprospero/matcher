"""Tests for US-75-009: video_id-to-metadata lookup for context access during matching.

Verifies that _build_video_metadata merges video_search_results and caption_results
into a single dict with title, description, tags, and chapters — defaulting to empty.
"""
import pytest
from unittest.mock import MagicMock
from src.stages.match import MatchStage


def _make_state(video_search_results=None, caption_results=None):
    """Create a minimal mock state with optional video_search_results and caption_results."""
    state = MagicMock()
    state.video_search_results = video_search_results or []
    state.caption_results = caption_results or {}
    return state


class TestBuildVideoMetadata:
    """Tests for MatchStage._build_video_metadata."""

    def test_metadata_from_caption_results_contains_all_keys(self):
        """video_metadata built from mock caption_results contains title, description, tags, chapters."""
        stage = MatchStage()
        caption_results = {
            'vid_abc': {
                'video_tags': ['nature', 'wildlife'],
                'video_chapters': [
                    {'title': 'Intro', 'start_time': 0, 'end_time': 30},
                    {'title': 'Main', 'start_time': 30, 'end_time': 120},
                ],
                'segments': [],
            }
        }
        state = _make_state(caption_results=caption_results)

        result = stage._build_video_metadata(state)

        assert 'vid_abc' in result
        meta = result['vid_abc']
        assert 'title' in meta
        assert 'description' in meta
        assert 'tags' in meta
        assert 'chapters' in meta
        assert meta['tags'] == ['nature', 'wildlife']
        assert len(meta['chapters']) == 2
        assert meta['chapters'][0]['title'] == 'Intro'

    def test_metadata_from_vsr_and_caption_results_merged(self):
        """video_search_results provide title/desc, caption_results provide chapters."""
        stage = MatchStage()
        vsr = MagicMock()
        vsr.video_id = 'vid_xyz'
        vsr.title = 'Great Video'
        vsr.description = 'A description'
        vsr.video_tags = ['tag1']

        caption_results = {
            'vid_xyz': {
                'video_tags': ['tag2'],
                'video_chapters': [{'title': 'Ch1', 'start_time': 0, 'end_time': 60}],
                'segments': [],
            }
        }
        state = _make_state(
            video_search_results=[vsr],
            caption_results=caption_results,
        )

        result = stage._build_video_metadata(state)

        meta = result['vid_xyz']
        assert meta['title'] == 'Great Video'
        assert meta['description'] == 'A description'
        # VSR already had tags, so they should be kept (non-empty)
        assert meta['tags'] == ['tag1']
        # Chapters come from caption_results
        assert len(meta['chapters']) == 1

    def test_video_with_no_caption_result_gets_empty_defaults(self):
        """A video present in VSR but absent from caption_results gets empty defaults."""
        stage = MatchStage()
        vsr = MagicMock()
        vsr.video_id = 'vid_only_vsr'
        vsr.title = 'Title Only'
        vsr.description = ''
        vsr.video_tags = []

        state = _make_state(video_search_results=[vsr], caption_results={})

        result = stage._build_video_metadata(state)

        meta = result['vid_only_vsr']
        assert meta['title'] == 'Title Only'
        assert meta['description'] == ''
        assert meta['tags'] == []
        assert meta['chapters'] == []

    def test_no_none_values_in_metadata(self):
        """Missing metadata fields default to empty, never None."""
        stage = MatchStage()
        # VSR with None values
        vsr = MagicMock()
        vsr.video_id = 'vid_nulls'
        vsr.title = None
        vsr.description = None
        vsr.video_tags = None

        caption_results = {
            'vid_nulls': {
                'video_tags': None,
                'video_chapters': None,
                'segments': [],
            }
        }
        state = _make_state(
            video_search_results=[vsr],
            caption_results=caption_results,
        )

        result = stage._build_video_metadata(state)

        meta = result['vid_nulls']
        assert meta['title'] == ''
        assert meta['description'] == ''
        assert meta['tags'] == []
        assert meta['chapters'] == []

    def test_caption_results_tags_fill_empty_vsr_tags(self):
        """If VSR has empty tags, caption_results tags fill the gap."""
        stage = MatchStage()
        vsr = MagicMock()
        vsr.video_id = 'vid_fill'
        vsr.title = 'Video'
        vsr.description = 'Desc'
        vsr.video_tags = []

        caption_results = {
            'vid_fill': {
                'video_tags': ['filled_tag'],
                'video_chapters': [],
                'segments': [],
            }
        }
        state = _make_state(
            video_search_results=[vsr],
            caption_results=caption_results,
        )

        result = stage._build_video_metadata(state)

        assert result['vid_fill']['tags'] == ['filled_tag']

    def test_dict_vsr_handled(self):
        """video_search_results as dicts (not objects) are handled correctly."""
        stage = MatchStage()
        vsr_dict = {
            'video_id': 'vid_dict',
            'title': 'Dict Title',
            'description': 'Dict Desc',
            'video_tags': ['dtag'],
        }
        state = _make_state(video_search_results=[vsr_dict])

        result = stage._build_video_metadata(state)

        meta = result['vid_dict']
        assert meta['title'] == 'Dict Title'
        assert meta['tags'] == ['dtag']
        assert meta['chapters'] == []

    def test_empty_state_returns_empty_metadata(self):
        """Empty state returns empty metadata dict."""
        stage = MatchStage()
        state = _make_state()

        result = stage._build_video_metadata(state)

        assert result == {}


class TestTieredMatcherChaptersAccessor:
    """Tests for TieredMatcher._get_video_chapters."""

    def test_get_video_chapters_returns_chapters(self):
        """_get_video_chapters returns chapters list from metadata."""
        from src.matching.tiered_matcher import TieredMatcher
        from src.utils import SRTSegment

        chapters = [{'title': 'Ch1', 'start_time': 0, 'end_time': 60}]
        metadata = {'vid_1': {'title': 'T', 'description': 'D', 'tags': [], 'chapters': chapters}}

        matcher = TieredMatcher(video_metadata=metadata)
        seg = SRTSegment(index=0, start_time=0, end_time=10, text='test', source_file='vid_1')
        result = matcher._get_video_chapters(seg)

        assert result == chapters

    def test_get_video_chapters_returns_empty_for_missing(self):
        """_get_video_chapters returns empty list for unknown video_id."""
        from src.matching.tiered_matcher import TieredMatcher
        from src.utils import SRTSegment

        matcher = TieredMatcher(video_metadata={})
        seg = SRTSegment(index=0, start_time=0, end_time=10, text='test', source_file='vid_unknown')
        result = matcher._get_video_chapters(seg)

        assert result == []

    def test_get_video_chapters_no_metadata(self):
        """_get_video_chapters returns empty list when video_metadata is None."""
        from src.matching.tiered_matcher import TieredMatcher
        from src.utils import SRTSegment

        matcher = TieredMatcher(video_metadata=None)
        seg = SRTSegment(index=0, start_time=0, end_time=10, text='test', source_file='vid_1')
        result = matcher._get_video_chapters(seg)

        assert result == []
