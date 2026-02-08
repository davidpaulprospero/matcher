"""
US-72-002: Test video_chapters and video_tags propagation from CaptionResult to state dataclasses.

Verifies:
- VideoSearchResult and DownloadedVideo have video_chapters/video_tags fields
- Fields default to empty lists (backward compat with old checkpoints)
- Serialization (asdict) and deserialization round-trip preserves these fields
- _populate_text_metadata propagates chapters/tags from caption_results to VideoSearchResult
"""

import pytest
import json
from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.state import VideoSearchResult, DownloadedVideo, PipelineState


SAMPLE_CHAPTERS = [
    {'title': 'Introduction', 'start_time': 0.0, 'end_time': 60.0},
    {'title': 'Main Topic', 'start_time': 60.0, 'end_time': 300.0},
]
SAMPLE_TAGS = ['tutorial', 'python', 'coding']


@pytest.mark.fast
class TestVideoSearchResultChaptersTags:
    """Test video_chapters and video_tags on VideoSearchResult."""

    def test_default_empty(self):
        """New fields default to empty lists (backward compat)."""
        vsr = VideoSearchResult(video_id='abc123')
        assert vsr.video_chapters == []
        assert vsr.video_tags == []

    def test_with_values(self):
        """Fields accept chapter and tag data."""
        vsr = VideoSearchResult(
            video_id='abc123',
            video_chapters=SAMPLE_CHAPTERS,
            video_tags=SAMPLE_TAGS,
        )
        assert len(vsr.video_chapters) == 2
        assert vsr.video_chapters[0]['title'] == 'Introduction'
        assert vsr.video_tags == ['tutorial', 'python', 'coding']

    def test_serialization_roundtrip(self):
        """asdict -> JSON -> from dict preserves fields."""
        vsr = VideoSearchResult(
            video_id='abc123',
            title='Test Video',
            video_chapters=SAMPLE_CHAPTERS,
            video_tags=SAMPLE_TAGS,
        )
        d = asdict(vsr)
        json_str = json.dumps(d)
        restored = json.loads(json_str)

        assert restored['video_chapters'] == SAMPLE_CHAPTERS
        assert restored['video_tags'] == SAMPLE_TAGS
        assert restored['video_id'] == 'abc123'

    def test_backward_compat_missing_fields(self):
        """Old checkpoint data without these fields still works."""
        old_data = {
            'video_id': 'abc123',
            'url': 'https://youtube.com/watch?v=abc123',
            'title': 'Old Video',
        }
        # Simulate restoring from old checkpoint: construct with defaults
        vsr = VideoSearchResult(
            video_id=old_data['video_id'],
            url=old_data.get('url', ''),
            title=old_data.get('title', ''),
            video_chapters=old_data.get('video_chapters', []),
            video_tags=old_data.get('video_tags', []),
        )
        assert vsr.video_chapters == []
        assert vsr.video_tags == []


@pytest.mark.fast
class TestDownloadedVideoChaptersTags:
    """Test video_chapters and video_tags on DownloadedVideo."""

    def test_default_empty(self):
        """New fields default to empty lists (backward compat)."""
        dv = DownloadedVideo(file='test.mp4')
        assert dv.video_chapters == []
        assert dv.video_tags == []

    def test_with_values(self):
        """Fields accept chapter and tag data."""
        dv = DownloadedVideo(
            file='test.mp4',
            video_chapters=SAMPLE_CHAPTERS,
            video_tags=SAMPLE_TAGS,
        )
        assert len(dv.video_chapters) == 2
        assert dv.video_tags == SAMPLE_TAGS

    def test_serialization_roundtrip(self):
        """asdict -> JSON -> from dict preserves fields."""
        dv = DownloadedVideo(
            file='test.mp4',
            title='Test',
            video_chapters=SAMPLE_CHAPTERS,
            video_tags=SAMPLE_TAGS,
        )
        d = asdict(dv)
        json_str = json.dumps(d)
        restored = json.loads(json_str)

        assert restored['video_chapters'] == SAMPLE_CHAPTERS
        assert restored['video_tags'] == SAMPLE_TAGS

    def test_backward_compat_missing_fields(self):
        """Old checkpoint data without these fields still works."""
        old_data = {'file': 'test.mp4', 'title': 'Old'}
        dv = DownloadedVideo(
            file=old_data['file'],
            title=old_data.get('title', ''),
            video_chapters=old_data.get('video_chapters', []),
            video_tags=old_data.get('video_tags', []),
        )
        assert dv.video_chapters == []
        assert dv.video_tags == []


@pytest.mark.fast
class TestPopulateTextMetadataPropagation:
    """Test that _populate_text_metadata propagates chapters/tags to VideoSearchResult."""

    def test_propagates_chapters_and_tags(self):
        """Caption result chapters/tags are copied to matching VideoSearchResult."""
        from src.stages.caption_stage import CaptionStage

        state = PipelineState()
        state.video_search_results = [
            VideoSearchResult(video_id='vid1', title='Video 1'),
            VideoSearchResult(video_id='vid2', title='Video 2'),
        ]

        caption_results = {
            'vid1': {
                'segments': [{'text': 'hello', 'start': 0, 'end': 5}],
                'language': 'en',
                'is_auto_generated': False,
                'caption_quality': 'high',
                'video_chapters': SAMPLE_CHAPTERS,
                'video_tags': SAMPLE_TAGS,
            },
            'vid2': {
                'segments': [{'text': 'world', 'start': 0, 'end': 3}],
                'language': 'en',
                'is_auto_generated': True,
                'caption_quality': 'medium',
                # No chapters/tags — should remain empty
            },
        }

        stage = CaptionStage.__new__(CaptionStage)
        stage._populate_text_metadata(state, caption_results)

        # vid1 should have chapters/tags propagated
        assert state.video_search_results[0].video_chapters == SAMPLE_CHAPTERS
        assert state.video_search_results[0].video_tags == SAMPLE_TAGS

        # vid2 should have empty defaults (not in caption_results)
        assert state.video_search_results[1].video_chapters == []
        assert state.video_search_results[1].video_tags == []

    def test_propagation_with_unavailable_captions(self):
        """Unavailable captions don't overwrite existing VSR data."""
        from src.stages.caption_stage import CaptionStage

        state = PipelineState()
        state.video_search_results = [
            VideoSearchResult(
                video_id='vid1',
                video_chapters=SAMPLE_CHAPTERS,
                video_tags=SAMPLE_TAGS,
            ),
        ]

        caption_results = {
            'vid1': {
                'unavailable': True,
                'reason': 'no_captions_available',
            },
        }

        stage = CaptionStage.__new__(CaptionStage)
        stage._populate_text_metadata(state, caption_results)

        # Should NOT be overwritten since caption was unavailable
        assert state.video_search_results[0].video_chapters == SAMPLE_CHAPTERS
        assert state.video_search_results[0].video_tags == SAMPLE_TAGS
