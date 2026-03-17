"""
Integration test for tag-based keyword boost in matching flow (US-127-003).

Tests that scoring with video tags that overlap voiceover keywords produces
higher confidence scores than scoring without tags.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.scoring import MatchScoring
from src.matching.llm_reranker import LLMReranker, LLMRerankerConfig, filter_and_score_tags
from src.utils import SRTSegment


def _make_segment(text: str, source_file: str = "") -> SRTSegment:
    """Create a segment with given text."""
    return SRTSegment(
        index=1,
        start_time=0.0,
        end_time=5.0,
        text=text,
        source_file=source_file,
    )


def _make_config(extract_video_tags: bool = True):
    """Create a minimal config mock."""
    config = Mock()
    matching = Mock()
    matching.multimodal_enabled = True
    matching.multimodal_weights = None
    matching.pool_normalization_enabled = True
    matching.chapter_matching_enabled = False
    matching.topic_mismatch_penalty = 0.15
    matching.broll_boost = 0.1
    matching.caption_quality_adjustment_enabled = True
    matching.caption_quality_high_boost = 0.05
    matching.caption_quality_low_penalty = 0.1
    matching.apply_timing_penalty = True
    matching.skip_llm_threshold = 0.85
    matching.language_confidence_penalty = 0.0
    matching.temporal_coherence_enabled = False
    matching.iterative_chapter_boost = 0.05
    matching.no_chapter_fallback_strategy = 'global'

    ce = Mock()
    ce.extract_video_tags = extract_video_tags
    ce.extract_video_chapters = False
    ce.enrich_with_description = False
    matching.context_enrichment = ce

    scoring = Mock()
    scoring.confidence_floor = 0.1
    scoring.low_confidence_warning_threshold = 0.15
    scoring.voiceover_context_calibration = False
    scoring.voiceover_context_boost_max = 0.05
    scoring.voiceover_context_penalty_max = 0.03
    scoring.context_richness_calibration = False
    scoring.context_richness_boost_max = 0.08
    scoring.context_richness_penalty_max = 0.05
    scoring.context_richness_title_weight = 0.25
    scoring.context_richness_description_weight = 0.25
    scoring.context_richness_tags_weight = 0.25
    scoring.context_richness_chapters_weight = 0.25
    scoring.chapter_match_confidence_min = 0.5
    matching.scoring = scoring

    config.matching = matching
    config.global_cache = Mock()
    config.global_cache.current_project_boost = 0.0
    return config


@pytest.mark.fast
class TestTagKeywordBoostIntegration:
    """Integration tests for tag-based keyword boost in scoring flow."""

    def test_matching_with_overlapping_tags_higher_confidence(self):
        """Matching with tags that overlap voiceover keywords produces higher confidence."""
        config = _make_config()
        scoring = MatchScoring(config)

        # Voiceover segment with specific keywords
        vo_seg = _make_segment(
            "The ancient Roman architecture still stands today in modern Rome"
        )

        # Video segment
        vid_seg = _make_segment(
            "Ancient Roman architecture preserved in Rome's historic center",
            source_file="dQw4w9WgXcQ"
        )

        # Tags that overlap with voiceover keywords
        tags = ['roman', 'architecture', 'rome', 'history', 'ancient']

        # Apply all adjustments
        adjusted_confidence, reasons, breakdown = scoring.apply_all_adjustments(
            confidence=0.70,
            vo_segment=vo_seg,
            video_segment=vid_seg,
            video_tags=tags,
        )

        # Check that tag_keyword_boost is in the breakdown
        tag_keyword_entries = [b for b in breakdown if b['component'] == 'tag_keyword_boost']
        assert len(tag_keyword_entries) == 1, "tag_keyword_boost should be in breakdown"

        # Verify boost was applied
        boost_entry = tag_keyword_entries[0]
        assert boost_entry['adjustment'] > 0, "tag_keyword_boost should be positive"

    def test_matching_without_tags_no_boost(self):
        """Matching without video tags produces no tag boost."""
        config = _make_config()
        scoring = MatchScoring(config)

        # Same voiceover segment
        vo_seg = _make_segment(
            "The ancient Roman architecture still stands today in modern Rome"
        )

        # Same video segment
        vid_seg = _make_segment(
            "Ancient Roman architecture preserved in Rome's historic center",
            source_file="dQw4w9WgXcQ"
        )

        # No tags (empty list)
        tags = []

        # Apply all adjustments
        adjusted_confidence, reasons, breakdown = scoring.apply_all_adjustments(
            confidence=0.70,
            vo_segment=vo_seg,
            video_segment=vid_seg,
            video_tags=tags,
        )

        # Check that tag_keyword_boost is NOT in the breakdown or has zero adjustment
        tag_keyword_entries = [b for b in breakdown if b['component'] == 'tag_keyword_boost']
        assert len(tag_keyword_entries) == 0 or tag_keyword_entries[0]['adjustment'] == 0, \
            "tag_keyword_boost should not apply when tags are empty"

    def test_matching_with_non_overlapping_tags_no_boost(self):
        """Matching with tags that don't overlap voiceover keywords produces no boost."""
        config = _make_config()
        scoring = MatchScoring(config)

        # Voiceover segment with specific keywords
        vo_seg = _make_segment(
            "The ancient Roman architecture still stands today in modern Rome"
        )

        # Video segment
        vid_seg = _make_segment(
            "Ancient Roman architecture preserved in Rome's historic center",
            source_file="dQw4w9WgXcQ"
        )

        # Tags that DON'T overlap voiceover keywords
        tags = ['cooking', 'pasta', 'recipe', 'kitchen']

        # Apply all adjustments
        adjusted_confidence, reasons, breakdown = scoring.apply_all_adjustments(
            confidence=0.70,
            vo_segment=vo_seg,
            video_segment=vid_seg,
            video_tags=tags,
        )

        # Check that tag_keyword_boost is NOT in the breakdown or has zero adjustment
        tag_keyword_entries = [b for b in breakdown if b['component'] == 'tag_keyword_boost']
        assert len(tag_keyword_entries) == 0 or tag_keyword_entries[0]['adjustment'] == 0, \
            "tag_keyword_boost should not apply when tags don't overlap"

    def test_tag_overlap_boost_integration(self):
        """Tag overlap boost (US-78-003) is applied in addition to tag keyword boost."""
        config = _make_config()
        scoring = MatchScoring(config)

        # Voiceover segment with specific keywords
        vo_seg = _make_segment(
            "The ancient Roman architecture still stands today in modern Rome"
        )

        # Video segment
        vid_seg = _make_segment(
            "Ancient Roman architecture preserved in Rome's historic center",
            source_file="dQw4w9WgXcQ"
        )

        # Tags that overlap voiceover keywords
        tags = ['roman', 'architecture', 'rome', 'history', 'ancient']

        # Apply all adjustments
        adjusted_confidence, reasons, breakdown = scoring.apply_all_adjustments(
            confidence=0.70,
            vo_segment=vo_seg,
            video_segment=vid_seg,
            video_tags=tags,
        )

        # Check that both tag_keyword_boost and tag_overlap are in the breakdown
        tag_keyword_entries = [b for b in breakdown if b['component'] == 'tag_keyword_boost']
        tag_overlap_entries = [b for b in breakdown if b['component'] == 'tag_overlap']

        assert len(tag_keyword_entries) == 1, "tag_keyword_boost should be in breakdown"
        assert len(tag_overlap_entries) == 1, "tag_overlap should be in breakdown"

        # Both should have positive adjustments
        assert tag_keyword_entries[0]['adjustment'] > 0
        assert tag_overlap_entries[0]['adjustment'] > 0

    def test_confidence_breakdown_includes_tag_entries(self):
        """Verify confidence_breakdown includes tag-based boost entries."""
        config = _make_config()
        scoring = MatchScoring(config)

        vo_seg = _make_segment("Technology drives modern innovation")
        vid_seg = _make_segment("Modern tech innovations", source_file="abc123")
        tags = ['technology', 'innovation']

        _, reasons, breakdown = scoring.apply_all_adjustments(
            confidence=0.70,
            vo_segment=vo_seg,
            video_segment=vid_seg,
            video_tags=tags,
        )

        # Collect all tag-related components
        tag_components = [b['component'] for b in breakdown if 'tag' in b['component'].lower()]

        # Should include tag_keyword_boost and tag_overlap
        assert 'tag_keyword_boost' in tag_components, "tag_keyword_boost should be in breakdown"
        assert 'tag_overlap' in tag_components, "tag_overlap should be in breakdown"

    def test_none_tags_no_boost(self):
        """None video_tags produces no boost."""
        config = _make_config()
        scoring = MatchScoring(config)

        vo_seg = _make_segment("The ancient Roman architecture still stands today")
        vid_seg = _make_segment("Roman ruins from the ancient empire")

        adjusted_confidence, reasons, breakdown = scoring.apply_all_adjustments(
            confidence=0.70,
            vo_segment=vo_seg,
            video_segment=vid_seg,
            video_tags=None,
        )

        # Should have no tag-related adjustments
        tag_keyword_entries = [b for b in breakdown if b['component'] == 'tag_keyword_boost']
        tag_overlap_entries = [b for b in breakdown if b['component'] == 'tag_overlap']
        assert len(tag_keyword_entries) == 0
        assert len(tag_overlap_entries) == 0


@pytest.mark.fast
class TestTagFilteringAndRelevance:
    """Tests for tag filtering and relevance scoring (US-134-003)."""

    def test_filter_generic_tags(self):
        """Generic/low-signal tags are filtered out."""
        from src.matching.llm_reranker import filter_and_score_tags

        tags = ['video', 'youtube', 'vlog', 'music', 'official', 'roman architecture', 'ancient history']
        result = filter_and_score_tags(tags)

        # Generic tags should be removed
        assert 'video' not in result
        assert 'youtube' not in result
        assert 'vlog' not in result
        assert 'music' not in result
        assert 'official' not in result
        # Specific tags should remain
        assert 'roman architecture' in result or 'ancient history' in result

    def test_relevance_scoring_title_match(self):
        """Tags matching title get higher relevance."""
        from src.matching.llm_reranker import filter_and_score_tags

        tags = ['cooking', 'pasta', 'recipe', 'italian food']
        title = "Best Italian Pasta Recipe"
        result = filter_and_score_tags(tags, title=title)

        # Tags in title should be prioritized
        assert 'pasta' in result
        assert 'recipe' in result or 'italian food' in result

    def test_relevance_scoring_description_match(self):
        """Tags matching description get relevance bonus."""
        from src.matching.llm_reranker import filter_and_score_tags

        tags = ['travel', 'beach', 'sunset', 'photography']
        title = "My Vacation"
        description = "Beach sunset photography tips"
        result = filter_and_score_tags(tags, title=title, description=description)

        # Tags in description should be prioritized
        assert 'beach' in result or 'sunset' in result or 'photography' in result

    def test_max_tags_limit(self):
        """Result respects max_tags limit."""
        from src.matching.llm_reranker import filter_and_score_tags

        tags = ['tag1', 'tag2', 'tag3', 'tag4', 'tag5', 'tag6', 'tag7']
        result = filter_and_score_tags(tags, max_tags=3)

        assert len(result) <= 3

    def test_empty_input_returns_empty(self):
        """Empty input returns empty list."""
        from src.matching.llm_reranker import filter_and_score_tags

        assert filter_and_score_tags([]) == []
        assert filter_and_score_tags(None) == []
        assert filter_and_score_tags(['video', 'youtube']) == []

    def test_multi_word_tags_preferred(self):
        """Multi-word tags are preferred (more specific)."""
        from src.matching.llm_reranker import filter_and_score_tags

        tags = ['food', 'cooking', 'italian cooking', 'recipe']
        result = filter_and_score_tags(tags)

        # Multi-word tags should be prioritized
        assert 'italian cooking' in result or 'recipe' in result

    def test_build_video_context_uses_filtered_tags(self):
        """_build_video_context uses filtered tags."""
        # Build context with mixed tags
        result = LLMReranker._build_video_context(
            title="Roman Architecture Tour",
            description="Explore ancient Roman buildings",
            tags=['video', 'youtube', 'roman architecture', 'ancient buildings', 'history'],
            chapters=[]
        )

        # Generic tags should not appear in context
        assert 'video' not in result
        assert 'youtube' not in result
        # Filtered tags should appear
        assert 'roman architecture' in result or 'ancient buildings' in result or 'history' in result
