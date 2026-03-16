"""
Tests for src/matching/context_pipeline.py - Unified context pipeline with intelligent fallback.

Coverage:
- ContextPipeline initialization with various data sources
- Fallback chain: full -> partial -> minimal -> no_context
- Graceful degradation when signals are unavailable
- Logging at each fallback level
- ContextEnrichment dataclass fields
- Adaptive weights computation
"""

import pytest
import logging
from src.matching.context_pipeline import (
    ContextPipeline,
    ContextEnrichment,
    ContextLevel,
    ContextPriorityWeightsConfig
)


class TestContextPipeline:
    """Test ContextPipeline class."""

    @pytest.mark.fast
    def test_init_empty(self):
        """Test initialization with no data."""
        pipeline = ContextPipeline()
        assert pipeline.video_metadata == {}
        assert pipeline.transcript_data == {}
        assert pipeline.visual_descriptions == {}

    @pytest.mark.fast
    def test_init_with_data(self):
        """Test initialization with video metadata."""
        video_metadata = {
            'vid1': {'title': 'Test Video', 'description': 'A test'}
        }
        pipeline = ContextPipeline(video_metadata=video_metadata)
        assert 'vid1' in pipeline.video_metadata

    @pytest.mark.fast
    def test_full_context_all_signals(self):
        """Test full context when all signals are available."""
        video_metadata = {
            'vid1': {
                'title': 'Python Tutorial',
                'description': 'Learn Python programming',
                'tags': ['python', 'programming', 'tutorial'],
                'chapters': [{'title': 'Introduction'}, {'title': 'Basics'}]
            }
        }
        transcript_data = {
            'vid1': [{'text': 'Welcome to the tutorial'}]
        }
        visual_descriptions = {
            'vid1': 'A person typing on a laptop'
        }

        pipeline = ContextPipeline(
            video_metadata=video_metadata,
            transcript_data=transcript_data,
            visual_descriptions=visual_descriptions
        )

        result = pipeline.get_enriched_context('vid1')

        assert result.level == ContextLevel.FULL
        assert result.title == 'Python Tutorial'
        assert result.description == 'Learn Python programming'
        assert result.tags == ['python', 'programming', 'tutorial']
        assert len(result.chapters) == 2
        assert 'title' in result.available_signals
        assert 'description' in result.available_signals
        assert 'tags' in result.available_signals
        assert 'chapters' in result.available_signals
        assert 'transcript' in result.available_signals
        assert 'visual' in result.available_signals
        assert result.context_string != ""
        assert result.fallback_triggered is None

    @pytest.mark.fast
    def test_partial_context_no_transcript(self):
        """Test partial context when transcript/visual not available but has tags."""
        video_metadata = {
            'vid1': {
                'title': 'Python Tutorial',
                'description': 'Learn Python programming',
                'tags': ['python', 'programming'],
                'chapters': [{'title': 'Introduction'}]
            }
        }

        pipeline = ContextPipeline(video_metadata=video_metadata)
        result = pipeline.get_enriched_context('vid1')

        # With tags and chapters, this is FULL context (more than just title+description)
        assert result.level == ContextLevel.FULL
        assert result.title == 'Python Tutorial'
        assert 'transcript' not in result.available_signals
        assert 'visual' not in result.available_signals
        assert result.fallback_triggered is None
        assert result.context_string != ""

    @pytest.mark.fast
    def test_minimal_context_title_description_only(self):
        """Test minimal context when only title and description available."""
        video_metadata = {
            'vid1': {
                'title': 'Python Tutorial',
                'description': 'Learn Python programming'
            }
        }

        pipeline = ContextPipeline(video_metadata=video_metadata)
        result = pipeline.get_enriched_context('vid1')

        # Title + description is PARTIAL (basic metadata beyond just title)
        assert result.level == ContextLevel.PARTIAL
        assert result.title == 'Python Tutorial'
        assert result.tags == []
        assert result.chapters == []
        assert result.fallback_triggered == ContextLevel.FULL

    @pytest.mark.fast
    def test_no_context_fallback(self):
        """Test no context when no metadata available."""
        video_metadata = {
            'vid1': {}
        }

        pipeline = ContextPipeline(video_metadata=video_metadata)
        result = pipeline.get_enriched_context('vid1')

        assert result.level == ContextLevel.NONE
        assert result.context_string == ""
        assert result.available_signals == []
        assert result.fallback_triggered == ContextLevel.MINIMAL

    @pytest.mark.fast
    def test_unknown_video_id(self):
        """Test fallback for unknown video ID."""
        pipeline = ContextPipeline()
        result = pipeline.get_enriched_context('unknown_id')

        assert result.level == ContextLevel.NONE
        assert result.context_string == ""
        assert result.fallback_triggered == ContextLevel.MINIMAL

    @pytest.mark.fast
    def test_graceful_degradation_maintains_functionality(self):
        """Test that fallback maintains core functionality - title/description always used when available."""
        # This test verifies that even at minimal level, we still get useful context
        video_metadata = {
            'vid1': {
                'title': 'Test Video',
                'description': 'Test description'
            }
        }

        pipeline = ContextPipeline(video_metadata=video_metadata)
        result = pipeline.get_enriched_context('vid1')

        # Even with fallback, we should have minimal context
        assert result.level in [ContextLevel.MINIMAL, ContextLevel.PARTIAL, ContextLevel.FULL]
        assert result.title == 'Test Video'
        assert result.description == 'Test description'
        # Core functionality maintained - context string is usable
        assert 'Test Video' in result.context_string

    @pytest.mark.fast
    def test_adaptive_weights_computation(self):
        """Test adaptive weights are computed correctly."""
        video_metadata = {
            'vid1': {
                'title': 'Test',
                'description': 'Test desc'
            }
        }

        pipeline = ContextPipeline(
            video_metadata=video_metadata,
            use_adaptive_weights=True
        )

        # With only title and description, weights should be boosted
        weights = pipeline._compute_adaptive_weights(
            title=True,
            description=True,
            tags=False,
            chapters=False,
            transcript=False,
            visual=False
        )

        # Weights should be boosted for available signals
        assert weights.title > 0.35  # Boosted from base 0.35
        assert weights.description > 0.30  # Boosted from base 0.30

    @pytest.mark.fast
    def test_context_stats(self):
        """Test get_context_stats method."""
        video_metadata = {
            'vid1': {'title': 'Test', 'tags': ['a', 'b'], 'chapters': [{}]},
            'vid2': {'title': 'Test2', 'tags': ['c']},
            'vid3': {'title': 'Test3'}  # No tags or chapters
        }
        transcript_data = {'vid1': [{}]}
        visual_descriptions = {'vid1': 'visual'}

        pipeline = ContextPipeline(
            video_metadata=video_metadata,
            transcript_data=transcript_data,
            visual_descriptions=visual_descriptions
        )

        stats = pipeline.get_context_stats()

        assert stats['total_videos'] == 3
        assert stats['videos_with_tags'] == 2
        assert stats['videos_with_chapters'] == 1
        assert stats['videos_with_transcripts'] == 1
        assert stats['videos_with_visual'] == 1


class TestContextEnrichment:
    """Test ContextEnrichment dataclass."""

    @pytest.mark.fast
    def test_defaults(self):
        """Test ContextEnrichment default values."""
        enrichment = ContextEnrichment(
            level=ContextLevel.NONE,
            context_string=""
        )

        assert enrichment.level == ContextLevel.NONE
        assert enrichment.context_string == ""
        assert enrichment.title == ""
        assert enrichment.description == ""
        assert enrichment.tags == []
        assert enrichment.chapters == []
        assert enrichment.transcript_segments == []
        assert enrichment.visual_description == ""
        assert enrichment.available_signals == []
        assert enrichment.fallback_triggered is None

    @pytest.mark.fast
    def test_full_enrichment(self):
        """Test ContextEnrichment with all fields."""
        enrichment = ContextEnrichment(
            level=ContextLevel.FULL,
            context_string="Test context",
            title="Test Title",
            description="Test Description",
            tags=["tag1", "tag2"],
            chapters=[{"title": "Chapter 1"}],
            transcript_segments=[{"text": "Hello"}],
            visual_description="Visual",
            available_signals=["title", "description"],
            fallback_triggered=None
        )

        assert enrichment.level == ContextLevel.FULL
        assert enrichment.context_string == "Test context"
        assert enrichment.title == "Test Title"
        assert len(enrichment.tags) == 2


class TestContextLevel:
    """Test ContextLevel enum."""

    @pytest.mark.fast
    def test_enum_values(self):
        """Test ContextLevel enum values."""
        assert ContextLevel.FULL.value == "full"
        assert ContextLevel.PARTIAL.value == "partial"
        assert ContextLevel.MINIMAL.value == "minimal"
        assert ContextLevel.NONE.value == "none"


class TestContextPriorityWeightsConfig:
    """Test ContextPriorityWeightsConfig dataclass."""

    @pytest.mark.fast
    def test_defaults(self):
        """Test default weights."""
        weights = ContextPriorityWeightsConfig()

        assert weights.title == 0.35
        assert weights.description == 0.30
        assert weights.tags == 0.20
        assert weights.chapters == 0.15

    @pytest.mark.fast
    def test_custom_weights(self):
        """Test custom weights."""
        weights = ContextPriorityWeightsConfig(
            title=0.4,
            description=0.3,
            tags=0.2,
            chapters=0.1
        )

        assert weights.title == 0.4
        assert weights.chapters == 0.1
