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
    # US-111-008: Multi-signal enrichment factors (defaults for backward compat)
    config.matching.context_enrichment.description_enrichment_factor = 0.3
    config.matching.context_enrichment.tags_enrichment_factor = 0.2
    config.matching.context_enrichment.chapters_enrichment_factor = 0.3
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


class TestEmbedChannelContextConfig:
    """Tests for embed_channel_context config option (US-95-008)."""

    @pytest.mark.fast
    def test_config_option_embed_channel_context_default_true(self):
        """Config option embed_channel_context defaults to True."""
        from src.config.sections.matching import ContextEnrichmentConfig
        config = ContextEnrichmentConfig()
        assert config.embed_channel_context is True

    @pytest.mark.fast
    def test_config_option_embed_channel_context_can_be_disabled(self):
        """Config option embed_channel_context can be set to False."""
        from src.config.sections.matching import ContextEnrichmentConfig
        config = ContextEnrichmentConfig(embed_channel_context=False)
        assert config.embed_channel_context is False


class TestChannelContextSearchRelevance:
    """Tests for channel context improving search relevance (US-95-008)."""

    @pytest.mark.fast
    def test_channel_context_same_creator_videos_cluster(self):
        """Videos from same channel should have similar embedding prefixes."""
        stage = _get_stage()

        # Two videos from same channel
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='vid1', title='Tutorial Part 1', channel='CodingChannel'),
                FakeVideoSearchResult(video_id='vid2', title='Tutorial Part 2', channel='CodingChannel'),
                FakeVideoSearchResult(video_id='vid3', title='Unrelated Video', channel='OtherChannel'),
            ]
        )
        caption_results = {
            'vid1': {'segments': [{'text': 'learning python basics', 'start': 0, 'end': 5}], 'language': 'en', 'caption_quality': 'high'},
            'vid2': {'segments': [{'text': 'advanced python tricks', 'start': 0, 'end': 5}], 'language': 'en', 'caption_quality': 'high'},
            'vid3': {'segments': [{'text': 'cooking recipe', 'start': 0, 'end': 5}], 'language': 'en', 'caption_quality': 'high'},
        }
        config = _make_config(embed_channel_context=True)

        stage._populate_text_metadata(state, caption_results, config)

        # Extract channel prefixes
        texts = {e['text']: e['embedding_text'] for e in state.text_metadata}
        prefix1 = texts['learning python basics']  # [CodingChannel | Tutorial Part 1]
        prefix2 = texts['advanced python tricks']  # [CodingChannel | Tutorial Part 2]
        prefix3 = texts['cooking recipe']  # [OtherChannel | Unrelated Video]

        # Same channel videos have same channel name
        assert 'CodingChannel' in prefix1
        assert 'CodingChannel' in prefix2
        assert prefix1.startswith('[CodingChannel |')
        assert prefix2.startswith('[CodingChannel |')
        # Different channel has different channel name
        assert 'OtherChannel' in prefix3
        assert prefix3.startswith('[OtherChannel |')

    @pytest.mark.fast
    def test_channel_context_differentiates_similar_titles_different_channels(self):
        """Same title from different channels should have different prefixes."""
        stage = _get_stage()

        # Same title but different channels
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='vid1', title='How To Cook', channel='ChefChannel'),
                FakeVideoSearchResult(video_id='vid2', title='How To Cook', channel='TechChannel'),
            ]
        )
        caption_results = {
            'vid1': {'segments': [{'text': 'make delicious pasta', 'start': 0, 'end': 5}], 'language': 'en', 'caption_quality': 'high'},
            'vid2': {'segments': [{'text': 'code a web app', 'start': 0, 'end': 5}], 'language': 'en', 'caption_quality': 'high'},
        }
        config = _make_config(embed_channel_context=True)

        stage._populate_text_metadata(state, caption_results, config)

        texts = {e['text']: e['embedding_text'] for e in state.text_metadata}
        prefix1 = texts['make delicious pasta'].split('] ')[0]
        prefix2 = texts['code a web app'].split('] ')[0]

        # Different channels = different prefixes
        assert 'ChefChannel' in prefix1
        assert 'TechChannel' in prefix2
        assert prefix1 != prefix2

    @pytest.mark.fast
    def test_channel_context_embedding_similarity_pattern(self):
        """Channel context makes same-creator videos more similar in embedding space."""
        # This test verifies the expected behavior: channel prefix creates a
        # similarity pattern that search algorithms can leverage
        stage = _get_stage()

        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='v1', title='Video A', channel='MyChannel'),
                FakeVideoSearchResult(video_id='v2', title='Video B', channel='MyChannel'),
                FakeVideoSearchResult(video_id='v3', title='Video C', channel='OtherChannel'),
            ]
        )
        caption_results = {
            'v1': {'segments': [{'text': 'content about topic x', 'start': 0, 'end': 5}], 'language': 'en', 'caption_quality': 'high'},
            'v2': {'segments': [{'text': 'content about topic x', 'start': 0, 'end': 5}], 'language': 'en', 'caption_quality': 'high'},
            'v3': {'segments': [{'text': 'content about topic x', 'start': 0, 'end': 5}], 'language': 'en', 'caption_quality': 'high'},
        }
        config = _make_config(embed_channel_context=True)

        stage._populate_text_metadata(state, caption_results, config)

        texts = [e['embedding_text'] for e in state.text_metadata]

        # All three have same content text, only channel differs
        # v1 and v2 should have same channel prefix (MyChannel)
        # v3 should have different channel prefix (OtherChannel)
        assert texts[0].startswith('[MyChannel | Video A]')
        assert texts[1].startswith('[MyChannel | Video B]')
        assert texts[2].startswith('[OtherChannel | Video C]')

        # Simulate similarity calculation: shared channel = higher similarity
        def simple_similarity(a: str, b: str) -> float:
            """Count shared words / total words."""
            words_a = set(a.lower().split())
            words_b = set(b.lower().split())
            return len(words_a & words_b) / len(words_a | words_b)

        # Same content, same channel = highest similarity
        sim_same_channel = simple_similarity(texts[0], texts[1])
        # Same content, different channel = lower similarity
        sim_diff_channel = simple_similarity(texts[0], texts[2])

        # Channel context creates measurable similarity difference
        assert sim_same_channel > sim_diff_channel, "Same-channel videos should be more similar"
