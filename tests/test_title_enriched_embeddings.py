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
    channel: str = ""  # US-95-008: Channel for embedding enrichment


@dataclass
class FakeState:
    text_metadata: List = field(default_factory=list)
    caption_results: dict = field(default_factory=dict)
    video_search_results: List = field(default_factory=list)
    video_ids: List[str] = field(default_factory=list)


def _make_config(title_enriched_embeddings: bool = True, chapter_enriched_embeddings: bool = True):
    """Build a minimal config mock with context_enrichment settings."""
    config = MagicMock()
    config.matching.context_enrichment.title_enriched_embeddings = title_enriched_embeddings
    config.matching.context_enrichment.chapter_enriched_embeddings = chapter_enriched_embeddings
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


class TestChapterEnrichedEmbeddings:
    """Tests for US-73-004: Chapter-enriched embedding text."""

    def _make_caption_results_with_chapters(self, video_id, segments, chapters):
        """Build caption_results with chapter data."""
        return {
            video_id: {
                'segments': segments,
                'language': 'en',
                'is_auto_generated': False,
                'caption_quality': 'high',
                'video_chapters': chapters,
            }
        }

    @pytest.mark.fast
    def test_embedding_text_with_chapter_title(self):
        """When chapter_enriched=True and segment has chapter, format is '[Title | Chapter] text'."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title='My Video')
            ]
        )
        caption_results = self._make_caption_results_with_chapters(
            'abc123',
            [{'text': 'hello world', 'start': 10, 'end': 15}],
            [{'title': 'Introduction', 'start_time': 0, 'end_time': 60}],
        )
        config = _make_config(title_enriched_embeddings=True, chapter_enriched_embeddings=True)

        stage._populate_text_metadata(state, caption_results, config)

        assert len(state.text_metadata) == 1
        entry = state.text_metadata[0]
        assert entry['embedding_text'] == '[My Video | Introduction] hello world'

    @pytest.mark.fast
    def test_embedding_text_without_chapter_falls_back(self):
        """When segment has no chapter (chapter_index=-1), falls back to '[Title] text'."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title='My Video')
            ]
        )
        # No chapters provided - all segments map to chapter_index=-1
        caption_results = self._make_caption_results_with_chapters(
            'abc123',
            [{'text': 'hello world', 'start': 10, 'end': 15}],
            [],  # No chapters
        )
        config = _make_config(title_enriched_embeddings=True, chapter_enriched_embeddings=True)

        stage._populate_text_metadata(state, caption_results, config)

        assert len(state.text_metadata) == 1
        entry = state.text_metadata[0]
        assert entry['embedding_text'] == '[My Video] hello world'

    @pytest.mark.fast
    def test_chapter_enrichment_disabled_uses_title_only(self):
        """When chapter_enriched=False, even with chapter data, uses '[Title] text'."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title='My Video')
            ]
        )
        caption_results = self._make_caption_results_with_chapters(
            'abc123',
            [{'text': 'hello world', 'start': 10, 'end': 15}],
            [{'title': 'Introduction', 'start_time': 0, 'end_time': 60}],
        )
        config = _make_config(title_enriched_embeddings=True, chapter_enriched_embeddings=False)

        stage._populate_text_metadata(state, caption_results, config)

        assert len(state.text_metadata) == 1
        entry = state.text_metadata[0]
        assert entry['embedding_text'] == '[My Video] hello world'

    @pytest.mark.fast
    def test_mixed_chapter_and_no_chapter_segments(self):
        """Segments with chapters get enriched, those without get title-only."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title='My Video')
            ]
        )
        caption_results = self._make_caption_results_with_chapters(
            'abc123',
            [
                {'text': 'in chapter', 'start': 10, 'end': 15},
                {'text': 'outside chapter', 'start': 200, 'end': 205},
            ],
            # Chapter only covers 0-60s, so second segment (200-205) is outside
            [{'title': 'Intro', 'start_time': 0, 'end_time': 60}],
        )
        config = _make_config(title_enriched_embeddings=True, chapter_enriched_embeddings=True)

        stage._populate_text_metadata(state, caption_results, config)

        assert len(state.text_metadata) == 2
        entries = {e['text']: e.get('embedding_text') for e in state.text_metadata}
        assert entries['in chapter'] == '[My Video | Intro] in chapter'
        assert entries['outside chapter'] == '[My Video] outside chapter'


class TestTitleTruncationUS126010:
    """Tests for US-126-010: Title truncation for long titles."""

    @pytest.mark.fast
    def test_long_title_truncated_at_100_chars(self):
        """Titles longer than 100 characters are truncated in embedding text."""
        stage = _get_stage()
        long_title = 'A' * 150  # 150 character title
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title=long_title)
            ]
        )
        caption_results = _make_caption_results('abc123', [
            {'text': 'hello world', 'start': 0, 'end': 5}
        ])
        config = _make_config(title_enriched_embeddings=True)

        stage._populate_text_metadata(state, caption_results, config)

        entry = state.text_metadata[0]
        # Should be truncated to 100 chars
        assert entry['embedding_text'].startswith('[AAAAAAAAAA')
        # The truncated title should be exactly 100 chars inside the brackets
        title_in_brackets = entry['embedding_text'].split(']')[0][1:]
        assert len(title_in_brackets) == 100

    @pytest.mark.fast
    def test_title_at_exactly_100_chars_not_truncated(self):
        """Titles exactly 100 characters are not truncated."""
        stage = _get_stage()
        exact_100_title = 'A' * 100
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title=exact_100_title)
            ]
        )
        caption_results = _make_caption_results('abc123', [
            {'text': 'hello world', 'start': 0, 'end': 5}
        ])
        config = _make_config(title_enriched_embeddings=True)

        stage._populate_text_metadata(state, caption_results, config)

        entry = state.text_metadata[0]
        title_in_brackets = entry['embedding_text'].split(']')[0][1:]
        assert len(title_in_brackets) == 100

    @pytest.mark.fast
    def test_short_title_not_truncated(self):
        """Titles shorter than 100 characters are not modified."""
        stage = _get_stage()
        short_title = 'Normal Title'
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title=short_title)
            ]
        )
        caption_results = _make_caption_results('abc123', [
            {'text': 'hello world', 'start': 0, 'end': 5}
        ])
        config = _make_config(title_enriched_embeddings=True)

        stage._populate_text_metadata(state, caption_results, config)

        entry = state.text_metadata[0]
        assert entry['embedding_text'] == '[Normal Title] hello world'

    @pytest.mark.fast
    def test_long_title_with_chapter_truncated(self):
        """Long titles are truncated even when chapter enrichment is enabled."""
        stage = _get_stage()
        long_title = 'B' * 120
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title=long_title)
            ]
        )
        caption_results = {
            'abc123': {
                'segments': [{'text': 'hello world', 'start': 10, 'end': 15}],
                'language': 'en',
                'is_auto_generated': False,
                'caption_quality': 'high',
                'video_chapters': [{'title': 'Intro', 'start_time': 0, 'end_time': 60}],
            }
        }
        config = _make_config(title_enriched_embeddings=True, chapter_enriched_embeddings=True)

        stage._populate_text_metadata(state, caption_results, config)

        entry = state.text_metadata[0]
        # The prefix before chapter is truncated to 100 chars
        # Format is [title | chapter]
        prefix = entry['embedding_text'].split(']')[0][1:]
        title_part = prefix.split(' | ')[0] if ' | ' in prefix else prefix
        assert len(title_part) == 100
        # Should still have chapter after title
        assert 'Intro' in entry['embedding_text']


class TestChannelEnrichedEmbeddingsUS95008:
    """Tests for US-95-008: Channel-enriched embedding text."""

    @pytest.mark.fast
    def test_embedding_text_with_channel(self):
        """When channel is present, format is '[Channel | Title] text'."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title='My Great Video', channel='TestChannel')
            ]
        )
        caption_results = _make_caption_results('abc123', [
            {'text': 'hello world', 'start': 0, 'end': 5}
        ])
        config = _make_config(title_enriched_embeddings=True)

        stage._populate_text_metadata(state, caption_results, config)

        assert len(state.text_metadata) == 1
        entry = state.text_metadata[0]
        # Format should be [channel | title] text
        assert entry['embedding_text'] == '[TestChannel | My Great Video] hello world'

    @pytest.mark.fast
    def test_embedding_text_with_channel_and_chapter(self):
        """When channel and chapter are present, format is '[Channel | Title | Chapter] text'."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title='My Great Video', channel='TestChannel')
            ]
        )
        caption_results = {
            'abc123': {
                'segments': [{'text': 'hello world', 'start': 10, 'end': 15}],
                'language': 'en',
                'is_auto_generated': False,
                'caption_quality': 'high',
                'video_chapters': [{'title': 'Intro', 'start_time': 0, 'end_time': 60}],
            }
        }
        config = _make_config(title_enriched_embeddings=True, chapter_enriched_embeddings=True)

        stage._populate_text_metadata(state, caption_results, config)

        assert len(state.text_metadata) == 1
        entry = state.text_metadata[0]
        assert entry['embedding_text'] == '[TestChannel | My Great Video | Intro] hello world'

    @pytest.mark.fast
    def test_embedding_text_channel_disabled(self):
        """When embed_channel_context=False, channel is not included."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                FakeVideoSearchResult(video_id='abc123', title='My Great Video', channel='TestChannel')
            ]
        )
        caption_results = _make_caption_results('abc123', [
            {'text': 'hello world', 'start': 0, 'end': 5}
        ])
        config = _make_config(title_enriched_embeddings=True)
        config.matching.context_enrichment.embed_channel_context = False

        stage._populate_text_metadata(state, caption_results, config)

        entry = state.text_metadata[0]
        # Should fallback to just title when channel enrichment is disabled
        assert entry['embedding_text'] == '[My Great Video] hello world'

    @pytest.mark.fast
    def test_embedding_text_dict_with_channel(self):
        """video_search_results dicts with channel field work correctly."""
        stage = _get_stage()
        state = FakeState(
            video_search_results=[
                {'video_id': 'abc123', 'title': 'Dict Title', 'channel': 'DictChannel'}
            ]
        )
        caption_results = _make_caption_results('abc123', [
            {'text': 'hello', 'start': 0, 'end': 5}
        ])
        config = _make_config(title_enriched_embeddings=True)

        stage._populate_text_metadata(state, caption_results, config)

        entry = state.text_metadata[0]
        assert entry['embedding_text'] == '[DictChannel | Dict Title] hello'
