"""
Tests for US-95-008: Channel-enriched embeddings.

Verifies that _populate_text_metadata correctly includes channel name in
embedding text when context_enrichment.embed_channel_context is enabled.
"""

import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass, field
from typing import List


@dataclass
class FakeVideoSearchResult:
    video_id: str = ""
    title: str = ""
    channel: str = ""
    url: str = ""


@dataclass
class FakeState:
    text_metadata: List = field(default_factory=list)
    caption_results: dict = field(default_factory=dict)
    video_search_results: List = field(default_factory=list)
    video_ids: List[str] = field(default_factory=list)


def _make_config(
    title_enriched_embeddings: bool = True,
    chapter_enriched_embeddings: bool = True,
    embed_channel_context: bool = True
):
    """Build a minimal config mock with context_enrichment settings."""
    config = MagicMock()
    config.matching.context_enrichment.title_enriched_embeddings = title_enriched_embeddings
    config.matching.context_enrichment.chapter_enriched_embeddings = chapter_enriched_embeddings
    config.matching.context_enrichment.embed_channel_context = embed_channel_context
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


class TestChannelEnrichedEmbeddings:
    """Tests for channel-enriched embedding text construction (US-95-008)."""

    @pytest.mark.fast
    def test_embedding_text_has_channel_prefix_when_enabled(self):
        """When embed_channel_context=True, format is '[Channel | Title] text'."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title='My Great Video', channel='TechChannel')
            ]
        )
        caption_results = _make_caption_results('abc123', [
            {'text': 'hello world', 'start': 0, 'end': 5}
        ])
        config = _make_config(embed_channel_context=True)

        stage._populate_text_metadata(state, caption_results, config)

        assert len(state.text_metadata) == 1
        entry = state.text_metadata[0]
        # Embedding text has channel and title prefix
        assert entry['embedding_text'] == '[TechChannel | My Great Video] hello world'

    @pytest.mark.fast
    def test_embedding_text_without_channel_when_disabled(self):
        """When embed_channel_context=False, format is '[Title] text' (no channel)."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title='My Great Video', channel='TechChannel')
            ]
        )
        caption_results = _make_caption_results('abc123', [
            {'text': 'hello world', 'start': 0, 'end': 5}
        ])
        config = _make_config(embed_channel_context=False)

        stage._populate_text_metadata(state, caption_results, config)

        assert len(state.text_metadata) == 1
        entry = state.text_metadata[0]
        # Embedding text has title only, no channel
        assert entry['embedding_text'] == '[My Great Video] hello world'

    @pytest.mark.fast
    def test_embedding_text_with_channel_and_chapter(self):
        """When channel + title + chapter enabled, format is '[Channel | Title | Chapter] text'."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title='My Video', channel='MyChannel')
            ]
        )
        caption_results = {
            'abc123': {
                'segments': [{'text': 'hello world', 'start': 10, 'end': 15}],
                'language': 'en',
                'is_auto_generated': False,
                'caption_quality': 'high',
                'video_chapters': [{'title': 'Introduction', 'start_time': 0, 'end_time': 60}],
            }
        }
        config = _make_config(embed_channel_context=True, chapter_enriched_embeddings=True)

        stage._populate_text_metadata(state, caption_results, config)

        assert len(state.text_metadata) == 1
        entry = state.text_metadata[0]
        assert entry['embedding_text'] == '[MyChannel | My Video | Introduction] hello world'

    @pytest.mark.fast
    def test_no_channel_in_prefix_when_channel_empty(self):
        """When channel is empty, falls back to '[Title] text' even if enabled."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title='My Video', channel='')
            ]
        )
        caption_results = _make_caption_results('abc123', [
            {'text': 'hello world', 'start': 0, 'end': 5}
        ])
        config = _make_config(embed_channel_context=True)

        stage._populate_text_metadata(state, caption_results, config)

        assert len(state.text_metadata) == 1
        entry = state.text_metadata[0]
        assert entry['embedding_text'] == '[My Video] hello world'

    @pytest.mark.fast
    def test_dict_search_results_with_channel(self):
        """video_search_results containing dicts (not dataclasses) work with channel."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                {'video_id': 'abc123', 'title': 'Dict Title', 'channel': 'DictChannel'}
            ]
        )
        caption_results = _make_caption_results('abc123', [
            {'text': 'hello', 'start': 0, 'end': 5}
        ])
        config = _make_config(embed_channel_context=True)

        stage._populate_text_metadata(state, caption_results, config)

        entry = state.text_metadata[0]
        assert entry['embedding_text'] == '[DictChannel | Dict Title] hello'

    @pytest.mark.fast
    def test_multiple_videos_get_correct_channels(self):
        """Each video's segments get the correct channel prefix."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='vid1', title='First Video', channel='ChannelOne'),
                FakeVideoSearchResult(video_id='vid2', title='Second Video', channel='ChannelTwo'),
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
        config = _make_config(embed_channel_context=True)

        stage._populate_text_metadata(state, caption_results, config)

        assert len(state.text_metadata) == 2
        texts = {e['text']: e.get('embedding_text') for e in state.text_metadata}
        assert texts['seg1 text'] == '[ChannelOne | First Video] seg1 text'
        assert texts['seg2 text'] == '[ChannelTwo | Second Video] seg2 text'

    @pytest.mark.fast
    def test_original_text_not_mutated(self):
        """Original text field must never contain channel prefix."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title='Title Here', channel='ChannelX')
            ]
        )
        caption_results = _make_caption_results('abc123', [
            {'text': 'original text', 'start': 0, 'end': 5},
        ])
        config = _make_config(embed_channel_context=True)

        stage._populate_text_metadata(state, caption_results, config)

        entry = state.text_metadata[0]
        # text must be the original, no prefix
        assert not entry['text'].startswith('[')
        # embedding_text must have the prefix
        assert entry['embedding_text'].startswith('[ChannelX | Title Here] ')
