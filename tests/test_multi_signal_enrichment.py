"""
Tests for US-111-008: Multi-Signal Embedding Context Enrichment.

Verifies that enrichment factors (description, tags, chapters) control
weighted combination of metadata signals in embedding text construction.
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
    chapter_enriched: bool = True,
    description_enriched: bool = True,
    channel_enriched: bool = True,
    description_enrichment_factor: float = 0.3,
    tags_enrichment_factor: float = 0.2,
    chapters_enrichment_factor: float = 0.3,
    max_keywords_from_description: int = 3,
):
    """Build a minimal config mock with context_enrichment settings."""
    config = MagicMock()
    config.matching.context_enrichment.title_enriched_embeddings = title_enriched
    config.matching.context_enrichment.chapter_enriched_embeddings = chapter_enriched
    config.matching.context_enrichment.description_enriched_embeddings = description_enriched
    config.matching.context_enrichment.embed_channel_context = channel_enriched
    # US-111-008: Multi-signal enrichment factors
    config.matching.context_enrichment.description_enrichment_factor = description_enrichment_factor
    config.matching.context_enrichment.tags_enrichment_factor = tags_enrichment_factor
    config.matching.context_enrichment.chapters_enrichment_factor = chapters_enrichment_factor
    # US-111-006: Enhanced keyword extraction settings
    config.matching.ngram_enabled = True
    config.matching.max_keywords_from_description = max_keywords_from_description
    return config


def _make_caption_results(
    video_id: str,
    segments: list,
    video_description: str = '',
    video_tags: list = None,
    video_chapters: list = None,
):
    """Build caption_results dict for a single video."""
    return {
        video_id: {
            'segments': segments,
            'language': 'en',
            'is_auto_generated': False,
            'caption_quality': 'high',
            'video_description': video_description,
            'video_tags': video_tags or [],
            'video_chapters': video_chapters or [],
        }
    }


def _get_stage():
    """Instantiate CaptionStage for testing."""
    from src.stages.caption_stage import CaptionStage
    return CaptionStage()


class TestMultiSignalEnrichmentFactors:
    """Tests for US-111-008: Multi-signal embedding context enrichment."""

    @pytest.mark.fast
    def test_tags_appended_when_tags_factor_gt_0(self):
        """When tags_enrichment_factor > 0, video tags are appended to embedding_text."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title='Wildlife Documentary')
            ]
        )
        video_tags = ['wildlife', 'nature', 'conservation', 'africa', 'safari']
        caption_results = _make_caption_results(
            'abc123',
            [{'text': 'elephants in the wild', 'start': 0, 'end': 5}],
            video_description='',
            video_tags=video_tags,
        )
        config = _make_config(
            title_enriched=True,
            description_enriched=False,
            tags_enrichment_factor=0.4,  # 40% of tags = 2 tags
        )

        stage._populate_text_metadata(state, caption_results, config)

        assert len(state.text_metadata) == 1
        entry = state.text_metadata[0]
        embed = entry['embedding_text']
        # Should contain [tags: ...] suffix
        assert '[tags:' in embed
        # With 0.4 factor and 5 tags, should get 2 tags (max(1, int(5 * 0.4)) = 2)
        assert 'wildlife' in embed
        assert 'nature' in embed
        # Should NOT have more than 2 tags
        assert embed.count('tags:') == 1

    @pytest.mark.fast
    def test_no_tags_when_tags_factor_is_zero(self):
        """When tags_enrichment_factor = 0, no tags are appended even if tags available."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title='My Video')
            ]
        )
        caption_results = _make_caption_results(
            'abc123',
            [{'text': 'hello world', 'start': 0, 'end': 5}],
            video_tags=['tag1', 'tag2', 'tag3'],
        )
        config = _make_config(
            title_enriched=True,
            tags_enrichment_factor=0.0,  # Disabled
        )

        stage._populate_text_metadata(state, caption_results, config)

        entry = state.text_metadata[0]
        embed = entry['embedding_text']
        # Should NOT contain tags
        assert '[tags:' not in embed

    @pytest.mark.fast
    def test_no_tags_when_tags_unavailable(self):
        """When tags are not available, no tags are appended even if factor > 0."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title='My Video')
            ]
        )
        caption_results = _make_caption_results(
            'abc123',
            [{'text': 'hello world', 'start': 0, 'end': 5}],
            video_tags=[],  # Empty tags
        )
        config = _make_config(
            title_enriched=True,
            tags_enrichment_factor=0.5,
        )

        stage._populate_text_metadata(state, caption_results, config)

        entry = state.text_metadata[0]
        embed = entry['embedding_text']
        # Should NOT contain tags when tags list is empty
        assert '[tags:' not in embed

    @pytest.mark.fast
    def test_chapters_included_when_chapters_factor_gt_0(self):
        """When chapters_enrichment_factor > 0, chapter title is included in prefix."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title='Documentary')
            ]
        )
        # Use chapters that will definitely map to first segment (start_time matches segment start)
        video_chapters = [
            {'title': 'Intro', 'start_time': 0, 'end_time': 60},
            {'title': 'Wildlife', 'start_time': 60, 'end_time': 120},
        ]
        caption_results = _make_caption_results(
            'abc123',
            [{'text': 'hello world', 'start': 5, 'end': 10}],  # In "Intro" chapter
            video_chapters=video_chapters,
        )
        config = _make_config(
            title_enriched=True,
            chapter_enriched=True,
            chapters_enrichment_factor=0.3,
        )

        stage._populate_text_metadata(state, caption_results, config)

        entry = state.text_metadata[0]
        embed = entry['embedding_text']
        # Should include chapter in prefix (Intro, since segment is at 5s)
        assert 'Intro' in embed
        assert embed.startswith('[Documentary | Intro]')

    @pytest.mark.fast
    def test_chapters_excluded_when_chapters_factor_is_zero(self):
        """When chapters_enrichment_factor = 0, chapter not in prefix even if chapter_enriched=True."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title='Documentary')
            ]
        )
        video_chapters = [
            {'title': 'Introduction', 'start_time': 0},
            {'title': 'Wildlife', 'start_time': 60},
        ]
        caption_results = _make_caption_results(
            'abc123',
            [{'text': 'hello world', 'start': 65, 'end': 70}],
            video_chapters=video_chapters,
        )
        config = _make_config(
            title_enriched=True,
            chapter_enriched=True,
            chapters_enrichment_factor=0.0,  # Disabled
        )

        stage._populate_text_metadata(state, caption_results, config)

        entry = state.text_metadata[0]
        embed = entry['embedding_text']
        # Should NOT include chapter in prefix
        assert embed == '[Documentary] hello world'

    @pytest.mark.fast
    def test_description_keywords_respect_description_factor(self):
        """Description keywords appended only when description_enrichment_factor > 0."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title='Wildlife')
            ]
        )
        caption_results = _make_caption_results(
            'abc123',
            [{'text': 'elephants', 'start': 0, 'end': 5}],
            video_description='wildlife conservation elephants lions africa safari',
        )
        # Factor > 0 but description_enriched is False - should NOT include desc
        config = _make_config(
            title_enriched=True,
            description_enriched=False,
            description_enrichment_factor=0.3,
        )
        config.matching.ngram_enabled = False

        stage._populate_text_metadata(state, caption_results, config)

        entry = state.text_metadata[0]
        embed = entry['embedding_text']
        # Should NOT contain desc since description_enriched is False
        assert '[desc:' not in embed

    @pytest.mark.fast
    def test_description_included_when_factor_and_flag_both_true(self):
        """Description keywords included when BOTH description_enriched=True AND factor > 0."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title='Wildlife')
            ]
        )
        caption_results = _make_caption_results(
            'abc123',
            [{'text': 'elephants', 'start': 0, 'end': 5}],
            video_description='wildlife conservation elephants lions africa safari',
        )
        config = _make_config(
            title_enriched=True,
            description_enriched=True,
            description_enrichment_factor=0.3,
        )
        config.matching.ngram_enabled = False

        stage._populate_text_metadata(state, caption_results, config)

        entry = state.text_metadata[0]
        embed = entry['embedding_text']
        # Should contain desc since both are enabled
        assert '[desc:' in embed

    @pytest.mark.fast
    def test_all_three_enrichment_signals_combined(self):
        """All three enrichment signals can be combined in embedding_text."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title='Nature Doc')
            ]
        )
        # Use chapters that map to the first segment (start matches)
        video_chapters = [
            {'title': 'Intro', 'start_time': 0, 'end_time': 60},
            {'title': 'Hunting', 'start_time': 60, 'end_time': 120},
        ]
        caption_results = _make_caption_results(
            'abc123',
            [{'text': 'lion hunting prey', 'start': 5, 'end': 10}],  # In "Intro" chapter
            video_description='africa wildlife safari lions',
            video_tags=['wildlife', 'nature', 'lions'],
            video_chapters=video_chapters,
        )
        config = _make_config(
            title_enriched=True,
            chapter_enriched=True,
            description_enriched=True,
            channel_enriched=False,
            description_enrichment_factor=0.3,
            tags_enrichment_factor=0.33,  # 1 tag from 3
            chapters_enrichment_factor=0.3,
        )
        config.matching.ngram_enabled = False

        stage._populate_text_metadata(state, caption_results, config)

        entry = state.text_metadata[0]
        embed = entry['embedding_text']
        # Should have all three signals:
        # 1. Chapter in prefix: [Nature Doc | Intro]
        assert 'Intro' in embed
        # 2. Description: [desc: ...]
        assert '[desc:' in embed
        # 3. Tags: [tags: ...]
        assert '[tags:' in embed

    @pytest.mark.fast
    def test_backward_compat_no_context_enrichment_config(self):
        """Backward compatibility: works when context_enrichment config is None."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title='Test')
            ]
        )
        caption_results = _make_caption_results(
            'abc123',
            [{'text': 'test', 'start': 0, 'end': 5}],
            video_description='test description',
            video_tags=['tag1'],
        )
        # Config with no context_enrichment attribute
        config = MagicMock()
        config.matching = None

        # Should not crash, should still produce embedding_text
        stage._populate_text_metadata(state, caption_results, config)

        assert len(state.text_metadata) == 1
        # Basic embedding text should still be generated from title

    @pytest.mark.fast
    def test_tags_factor_scales_num_tags(self):
        """Higher tags_enrichment_factor includes more tags."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title='Test')
            ]
        )
        video_tags = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']  # 10 tags
        caption_results = _make_caption_results(
            'abc123',
            [{'text': 'test', 'start': 0, 'end': 5}],
            video_tags=video_tags,
        )
        # With 0.2 factor: max(1, int(10 * 0.2)) = 2 tags
        config = _make_config(tags_enrichment_factor=0.2)

        stage._populate_text_metadata(state, caption_results, config)

        entry = state.text_metadata[0]
        embed = entry['embedding_text']
        # Count tags in the [tags: ...] section
        import re
        match = re.search(r'\[tags: ([^\]]+)\]', embed)
        assert match is not None
        tags_in_embed = match.group(1).split()
        assert len(tags_in_embed) == 2


class TestContextEnrichmentConfigValidation:
    """Tests for ContextEnrichmentConfig validation of enrichment factors."""

    @pytest.mark.fast
    def test_enrichment_factors_default_values(self):
        """Default enrichment factors match story requirements."""
        from src.config.sections.matching import ContextEnrichmentConfig
        config = ContextEnrichmentConfig()
        assert config.description_enrichment_factor == 0.3
        assert config.tags_enrichment_factor == 0.2
        assert config.chapters_enrichment_factor == 0.3

    @pytest.mark.fast
    def test_enrichment_factor_out_of_range_raises(self):
        """Enrichment factors outside 0-1 range raise ValueError."""
        from src.config.sections.matching import ContextEnrichmentConfig

        with pytest.raises(ValueError):
            ContextEnrichmentConfig(description_enrichment_factor=-0.1)

        with pytest.raises(ValueError):
            ContextEnrichmentConfig(tags_enrichment_factor=1.5)

    @pytest.mark.fast
    def test_enrichment_factor_at_boundaries_valid(self):
        """Enrichment factors at 0 and 1 are valid."""
        from src.config.sections.matching import ContextEnrichmentConfig
        # Should not raise
        config = ContextEnrichmentConfig(
            description_enrichment_factor=0.0,
            tags_enrichment_factor=1.0,
            chapters_enrichment_factor=0.5,
        )
        assert config.description_enrichment_factor == 0.0
        assert config.tags_enrichment_factor == 1.0
        assert config.chapters_enrichment_factor == 0.5
