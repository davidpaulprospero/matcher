"""
Tests for US-70-008: Title-enriched embeddings.

Verifies that _populate_text_metadata correctly prepends video title to
embedding text when context_enrichment.title_enriched_embeddings is enabled,
while preserving the original caption text for display/matching.
"""

import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass, field
from typing import List


@dataclass
class FakeVideoSearchResult:
    video_id: str = ""
    title: str = ""
    url: str = ""


@dataclass
class FakeState:
    text_metadata: List = field(default_factory=list)
    caption_results: dict = field(default_factory=dict)
    video_search_results: List = field(default_factory=list)
    video_ids: List[str] = field(default_factory=list)


def _make_config(title_enriched_embeddings: bool = True):
    """Build a minimal config mock with context_enrichment settings."""
    config = MagicMock()
    config.matching.context_enrichment.title_enriched_embeddings = title_enriched_embeddings
    return config


def _make_caption_results(video_id: str, segments: list):
    """Build caption_results dict for a single video."""
    return {
        video_id: {
            'segments': segments,
            'language': 'en',
            'is_auto_generated': False,
            'caption_quality': 'high',
        }
    }


def _get_stage():
    """Instantiate CaptionStage for testing."""
    from src.stages.caption_stage import CaptionStage
    return CaptionStage()


class TestTitleEnrichedEmbeddings:
    """Tests for title-enriched embedding text construction."""

    @pytest.mark.fast
    def test_embedding_text_has_title_prefix_when_enabled(self):
        """When title_enriched_embeddings=True, embedding_text has '[Title] text' format."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title='My Great Video')
            ]
        )
        caption_results = _make_caption_results('abc123', [
            {'text': 'hello world', 'start': 0, 'end': 5}
        ])
        config = _make_config(title_enriched_embeddings=True)

        stage._populate_text_metadata(state, caption_results, config)

        assert len(state.text_metadata) == 1
        entry = state.text_metadata[0]
        # Original text preserved
        assert entry['text'] == 'hello world'
        # Embedding text has title prefix
        assert entry['embedding_text'] == '[My Great Video] hello world'

    @pytest.mark.fast
    def test_no_embedding_text_when_disabled(self):
        """When title_enriched_embeddings=False, no embedding_text field added."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title='My Great Video')
            ]
        )
        caption_results = _make_caption_results('abc123', [
            {'text': 'hello world', 'start': 0, 'end': 5}
        ])
        config = _make_config(title_enriched_embeddings=False)

        stage._populate_text_metadata(state, caption_results, config)

        assert len(state.text_metadata) == 1
        entry = state.text_metadata[0]
        assert entry['text'] == 'hello world'
        assert 'embedding_text' not in entry

    @pytest.mark.fast
    def test_no_embedding_text_when_no_config(self):
        """When config is None, no embedding_text field added (backward compat)."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title='My Great Video')
            ]
        )
        caption_results = _make_caption_results('abc123', [
            {'text': 'hello world', 'start': 0, 'end': 5}
        ])

        stage._populate_text_metadata(state, caption_results, config=None)

        assert len(state.text_metadata) == 1
        entry = state.text_metadata[0]
        assert entry['text'] == 'hello world'
        assert 'embedding_text' not in entry

    @pytest.mark.fast
    def test_no_embedding_text_when_title_empty(self):
        """When video title is empty, no embedding_text field added."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title='')
            ]
        )
        caption_results = _make_caption_results('abc123', [
            {'text': 'hello world', 'start': 0, 'end': 5}
        ])
        config = _make_config(title_enriched_embeddings=True)

        stage._populate_text_metadata(state, caption_results, config)

        assert len(state.text_metadata) == 1
        entry = state.text_metadata[0]
        assert entry['text'] == 'hello world'
        assert 'embedding_text' not in entry

    @pytest.mark.fast
    def test_no_embedding_text_when_no_search_results(self):
        """When video_search_results is empty, no embedding_text even if enabled."""
        stage = _get_stage()
        state = FakeState(video_search_results=[])
        caption_results = _make_caption_results('abc123', [
            {'text': 'hello world', 'start': 0, 'end': 5}
        ])
        config = _make_config(title_enriched_embeddings=True)

        stage._populate_text_metadata(state, caption_results, config)

        assert len(state.text_metadata) == 1
        entry = state.text_metadata[0]
        assert 'embedding_text' not in entry

    @pytest.mark.fast
    def test_multiple_videos_get_correct_titles(self):
        """Each video's segments get the correct title prefix."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='vid1', title='First Video'),
                FakeVideoSearchResult(video_id='vid2', title='Second Video'),
            ]
        )
        caption_results = {
            'vid1': {
                'segments': [{'text': 'seg1 text', 'start': 0, 'end': 5}],
                'language': 'en',
                'caption_quality': 'high',
            },
            'vid2': {
                'segments': [{'text': 'seg2 text', 'start': 0, 'end': 5}],
                'language': 'en',
                'caption_quality': 'high',
            },
        }
        config = _make_config(title_enriched_embeddings=True)

        stage._populate_text_metadata(state, caption_results, config)

        assert len(state.text_metadata) == 2
        texts = {e['text']: e.get('embedding_text') for e in state.text_metadata}
        assert texts['seg1 text'] == '[First Video] seg1 text'
        assert texts['seg2 text'] == '[Second Video] seg2 text'

    @pytest.mark.fast
    def test_original_text_not_mutated(self):
        """Original text field must never contain title prefix."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title='Title Here')
            ]
        )
        caption_results = _make_caption_results('abc123', [
            {'text': 'original text', 'start': 0, 'end': 5},
            {'text': 'another segment', 'start': 5, 'end': 10},
        ])
        config = _make_config(title_enriched_embeddings=True)

        stage._populate_text_metadata(state, caption_results, config)

        for entry in state.text_metadata:
            # text must be the original, no title prefix
            assert not entry['text'].startswith('[')
            # embedding_text must have the prefix
            assert entry['embedding_text'].startswith('[Title Here] ')

    @pytest.mark.fast
    def test_dict_search_results_handled(self):
        """video_search_results containing dicts (not dataclasses) still work."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                {'video_id': 'abc123', 'title': 'Dict Title'}
            ]
        )
        caption_results = _make_caption_results('abc123', [
            {'text': 'hello', 'start': 0, 'end': 5}
        ])
        config = _make_config(title_enriched_embeddings=True)

        stage._populate_text_metadata(state, caption_results, config)

        entry = state.text_metadata[0]
        assert entry['embedding_text'] == '[Dict Title] hello'

    @pytest.mark.fast
    def test_skipped_unavailable_captions_not_enriched(self):
        """Unavailable/errored captions are skipped, not enriched."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title='Some Title')
            ]
        )
        caption_results = {
            'abc123': {
                'unavailable': True,
                'segments': [],
            }
        }
        config = _make_config(title_enriched_embeddings=True)

        stage._populate_text_metadata(state, caption_results, config)

        assert len(state.text_metadata) == 0
