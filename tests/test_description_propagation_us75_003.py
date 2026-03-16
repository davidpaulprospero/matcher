"""
Tests for US-75-003: Propagate video_description from caption_results to text_metadata.

Verifies that _populate_text_metadata correctly copies video_description from
CaptionResult dicts into each text_metadata entry, with backward-compatible
defaults when the field is missing (old cache data).
"""

import pytest
from dataclasses import dataclass, field
from typing import List

from src.stages.caption_stage import CaptionStage


@dataclass
class FakeState:
    text_metadata: List = field(default_factory=list)
    caption_results: dict = field(default_factory=dict)
    video_search_results: List = field(default_factory=list)
    video_ids: List[str] = field(default_factory=list)


def _make_caption_results(video_id: str, segments: list, video_description: str = ""):
    """Build caption_results dict for a single video with optional description."""
    result = {
        video_id: {
            'segments': segments,
            'language': 'en',
            'is_auto_generated': False,
            'caption_quality': 'high',
        }
    }
    if video_description:
        result[video_id]['video_description'] = video_description
    return result


class TestDescriptionPropagation:
    """Tests for video_description propagation from caption_results to text_metadata."""

    @pytest.mark.fast
    def test_description_propagated_to_text_metadata(self):
        """video_description from CaptionResult appears in text_metadata entries."""
        stage = CaptionStage()
        state = FakeState()
        caption_results = _make_caption_results(
            'vid1',
            [{'text': 'segment one', 'start': 0, 'end': 5}],
            video_description='A documentary about wildlife conservation'
        )

        stage._populate_text_metadata(state, caption_results)

        assert len(state.text_metadata) == 1
        entry = state.text_metadata[0]
        assert entry['video_description'] == 'A documentary about wildlife conservation'

    @pytest.mark.fast
    def test_description_present_on_all_segments(self):
        """All segments from same video share the same video_description."""
        stage = CaptionStage()
        state = FakeState()
        caption_results = _make_caption_results(
            'vid2',
            [
                {'text': 'seg A', 'start': 0, 'end': 3},
                {'text': 'seg B', 'start': 3, 'end': 6},
                {'text': 'seg C', 'start': 6, 'end': 9},
            ],
            video_description='Travel vlog through Japan'
        )

        stage._populate_text_metadata(state, caption_results)

        assert len(state.text_metadata) == 3
        for entry in state.text_metadata:
            assert entry['video_description'] == 'Travel vlog through Japan'

    @pytest.mark.fast
    def test_missing_description_defaults_to_empty_string(self):
        """When CaptionResult has no video_description (old cache), defaults to ''."""
        stage = CaptionStage()
        state = FakeState()
        # No video_description key at all — simulates old cache data
        caption_results = {
            'vid_old': {
                'segments': [{'text': 'old data', 'start': 0, 'end': 5}],
                'language': 'en',
                'is_auto_generated': False,
                'caption_quality': 'medium',
            }
        }

        stage._populate_text_metadata(state, caption_results)

        assert len(state.text_metadata) == 1
        entry = state.text_metadata[0]
        assert entry['video_description'] == ''

    @pytest.mark.fast
    def test_description_coexists_with_existing_fields(self):
        """video_description doesn't interfere with title, chapter, or other metadata."""
        stage = CaptionStage()
        state = FakeState()
        caption_results = _make_caption_results(
            'vid3',
            [{'text': 'test segment', 'start': 0, 'end': 5}],
            video_description='Cooking tutorial'
        )

        stage._populate_text_metadata(state, caption_results)

        entry = state.text_metadata[0]
        # Core fields still present
        assert entry['text'] == 'test segment'
        assert entry['video_path'] == 'vid3'
        assert entry['source_file'] == 'vid3'
        assert entry['caption_quality'] == 'high'
        # Description field present
        assert entry['video_description'] == 'Cooking tutorial'

    @pytest.mark.fast
    def test_multiple_videos_each_get_own_description(self):
        """Different videos propagate their own descriptions independently."""
        stage = CaptionStage()
        state = FakeState()
        caption_results = {
            'vidA': {
                'segments': [{'text': 'alpha', 'start': 0, 'end': 3}],
                'language': 'en',
                'is_auto_generated': False,
                'caption_quality': 'high',
                'video_description': 'Description for A',
            },
            'vidB': {
                'segments': [{'text': 'beta', 'start': 0, 'end': 3}],
                'language': 'en',
                'is_auto_generated': False,
                'caption_quality': 'medium',
                'video_description': 'Description for B',
            },
        }

        stage._populate_text_metadata(state, caption_results)

        assert len(state.text_metadata) == 2
        descriptions = {e['video_path']: e['video_description'] for e in state.text_metadata}
        assert descriptions['vidA'] == 'Description for A'
        assert descriptions['vidB'] == 'Description for B'
