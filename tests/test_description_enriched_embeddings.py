"""
Tests for US-75-011: Description keyword-enriched embeddings.

Verifies that _populate_text_metadata appends top description keywords
to embedding_text when description_enriched_embeddings is enabled,
and leaves embedding_text unchanged when disabled (backward compat).
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


def _make_config(
    title_enriched: bool = True,
    chapter_enriched: bool = False,
    description_enriched: bool = False,
):
    """Build a minimal config mock with context_enrichment settings."""
    config = MagicMock()
    config.matching.context_enrichment.title_enriched_embeddings = title_enriched
    config.matching.context_enrichment.chapter_enriched_embeddings = chapter_enriched
    config.matching.context_enrichment.description_enriched_embeddings = description_enriched
    return config


def _make_caption_results(video_id: str, segments: list, video_description: str = ''):
    """Build caption_results dict for a single video."""
    return {
        video_id: {
            'segments': segments,
            'language': 'en',
            'is_auto_generated': False,
            'caption_quality': 'high',
            'video_description': video_description,
        }
    }


def _get_stage():
    """Instantiate CaptionStage for testing."""
    from src.stages.caption_stage import CaptionStage
    return CaptionStage()


class TestDescriptionEnrichedEmbeddings:
    """Tests for US-75-011: Description keyword-enriched embedding text."""

    @pytest.mark.fast
    def test_embedding_text_includes_description_keywords_when_enabled(self):
        """When description_enriched_embeddings=True, embedding_text has appended keywords."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title='Wildlife Documentary')
            ]
        )
        description = (
            'This documentary explores wildlife conservation in Africa. '
            'Learn about elephants, lions, and conservation efforts to protect endangered species.'
        )
        caption_results = _make_caption_results(
            'abc123',
            [{'text': 'hello world', 'start': 0, 'end': 5}],
            video_description=description,
        )
        config = _make_config(title_enriched=True, description_enriched=True)

        stage._populate_text_metadata(state, caption_results, config)

        assert len(state.text_metadata) == 1
        entry = state.text_metadata[0]
        embed = entry['embedding_text']
        # Should start with title prefix
        assert embed.startswith('[Wildlife Documentary]')
        # Should contain [desc: ...] suffix with top 3 keywords
        assert '[desc:' in embed
        # Extract the desc bracket content
        desc_part = embed.split('[desc: ')[1].rstrip(']')
        desc_keywords = desc_part.split()
        assert len(desc_keywords) <= 3  # Top 3 keywords max
        # Original text preserved
        assert entry['text'] == 'hello world'

    @pytest.mark.fast
    def test_embedding_text_unchanged_when_description_disabled(self):
        """When description_enriched_embeddings=False, no description keywords appended."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title='My Video')
            ]
        )
        caption_results = _make_caption_results(
            'abc123',
            [{'text': 'hello world', 'start': 0, 'end': 5}],
            video_description='wildlife conservation elephants lions',
        )
        config = _make_config(title_enriched=True, description_enriched=False)

        stage._populate_text_metadata(state, caption_results, config)

        assert len(state.text_metadata) == 1
        entry = state.text_metadata[0]
        assert entry['embedding_text'] == '[My Video] hello world'

    @pytest.mark.fast
    def test_no_keywords_appended_when_description_empty(self):
        """When video_description is empty, no keywords appended even if enabled."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title='My Video')
            ]
        )
        caption_results = _make_caption_results(
            'abc123',
            [{'text': 'hello world', 'start': 0, 'end': 5}],
            video_description='',
        )
        config = _make_config(title_enriched=True, description_enriched=True)

        stage._populate_text_metadata(state, caption_results, config)

        entry = state.text_metadata[0]
        assert entry['embedding_text'] == '[My Video] hello world'

    @pytest.mark.fast
    def test_stop_words_filtered_from_keywords(self):
        """Stop words and YouTube-specific words are not included in keywords."""
        stage = _get_stage()
        keywords = stage._extract_description_keywords(
            'Subscribe to the channel and like this video about wildlife conservation'
        )
        # 'subscribe', 'the', 'channel', 'like', 'this', 'video', 'and', 'about' are stop words
        assert 'subscribe' not in keywords
        assert 'the' not in keywords
        assert 'channel' not in keywords
        assert 'like' not in keywords
        assert 'video' not in keywords
        # Content words should be present
        assert 'wildlife' in keywords
        assert 'conservation' in keywords

    @pytest.mark.fast
    def test_max_5_keywords_by_default(self):
        """At most 5 keywords are extracted by default."""
        stage = _get_stage()
        description = (
            'elephants lions giraffes zebras rhinos hippos crocodiles '
            'gazelles cheetahs leopards hyenas buffalo'
        )
        keywords = stage._extract_description_keywords(description)
        assert len(keywords) <= 5

    @pytest.mark.fast
    def test_keywords_sorted_by_frequency(self):
        """Keywords with higher frequency appear first."""
        stage = _get_stage()
        description = (
            'conservation conservation conservation '
            'wildlife wildlife '
            'elephants'
        )
        keywords = stage._extract_description_keywords(description)
        assert keywords[0] == 'conservation'
        assert keywords[1] == 'wildlife'
        assert keywords[2] == 'elephants'

    @pytest.mark.fast
    def test_short_words_filtered(self):
        """Words shorter than 3 characters are excluded."""
        stage = _get_stage()
        keywords = stage._extract_description_keywords('an ox in a zoo with big cat')
        # 'an', 'ox', 'in', 'a' are <3 chars; 'zoo', 'big', 'cat' are 3 chars
        assert 'ox' not in keywords
        assert 'an' not in keywords
        assert 'zoo' in keywords
        assert 'big' in keywords
        assert 'cat' in keywords

    @pytest.mark.fast
    def test_youtube_specific_words_filtered(self):
        """YouTube-specific words like subscribe, channel, link are filtered."""
        stage = _get_stage()
        keywords = stage._extract_description_keywords(
            'subscribe click link description instagram twitter facebook tiktok '
            'patreon merch discount photography'
        )
        for yt_word in ['subscribe', 'click', 'link', 'description', 'instagram',
                        'twitter', 'facebook', 'tiktok', 'patreon', 'merch', 'discount']:
            assert yt_word not in keywords
        assert 'photography' in keywords

    @pytest.mark.fast
    def test_empty_description_returns_empty_list(self):
        """Empty or None-like description returns no keywords."""
        stage = _get_stage()
        assert stage._extract_description_keywords('') == []
        assert stage._extract_description_keywords('   ') == []

    @pytest.mark.fast
    def test_description_with_only_stop_words_returns_empty(self):
        """Description with only stop/YouTube words returns empty list."""
        stage = _get_stage()
        keywords = stage._extract_description_keywords(
            'subscribe to the channel and like this video'
        )
        assert keywords == []

    @pytest.mark.fast
    def test_integration_uses_top_3_keywords_with_desc_prefix(self):
        """Integration: _populate_text_metadata appends top 3 description keywords with [desc:] prefix."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title='Nature Film')
            ]
        )
        description = (
            'conservation conservation conservation conservation '
            'wildlife wildlife wildlife '
            'elephants elephants '
            'giraffes '
            'rhinos'
        )
        caption_results = _make_caption_results(
            'abc123',
            [{'text': 'hello world', 'start': 0, 'end': 5}],
            video_description=description,
        )
        config = _make_config(title_enriched=True, description_enriched=True)

        stage._populate_text_metadata(state, caption_results, config)

        entry = state.text_metadata[0]
        embed = entry['embedding_text']
        # Should contain exactly [desc: kw1 kw2 kw3] with top 3 by frequency
        assert '[desc: conservation wildlife elephants]' in embed
        # Should NOT contain 4th or 5th keywords
        assert 'giraffes' not in embed
        assert 'rhinos' not in embed

    @pytest.mark.fast
    def test_extract_description_keywords_returns_frequency_ordered(self):
        """_extract_description_keywords returns keywords ordered by frequency (reusable extraction)."""
        stage = _get_stage()
        keywords = stage._extract_description_keywords(
            'photography photography landscape landscape landscape travel',
            max_keywords=3,
        )
        assert keywords == ['landscape', 'photography', 'travel']
