"""Tests for US-141-010: Visual-textual context fusion scoring."""

import pytest
from src.matching.scoring import compute_visual_text_fusion_score, _compute_text_similarity


class TestVisualTextFusionScore:
    """Test compute_visual_text_fusion_score function."""

    def test_fusion_enabled_both_signals(self):
        """Test fusion with both visual and text signals available."""
        visual_description = "A person walking on a beach at sunset"
        text_metadata = {
            "title": "Beach Sunset Walk",
            "description": "Beautiful evening walk on the sandy beach",
            "tags": ["sunset", "beach", "walking", "nature"]
        }
        vo_segment = "Take a peaceful walk along the beach at sunset"

        score, reason, components = compute_visual_text_fusion_score(
            visual_description=visual_description,
            text_metadata=text_metadata,
            vo_segment=vo_segment,
            visual_text_weight=0.20,
            fusion_enabled=True,
        )

        assert score >= 0.0
        assert score <= 1.0
        assert "fusion" in reason
        assert components['has_visual'] is True
        assert components['has_text'] is True

    def test_fusion_visual_only(self):
        """Test fusion with only visual signal available."""
        visual_description = "A person running in a park"
        text_metadata = {}  # Empty - no text metadata
        vo_segment = "Running through the park"

        score, reason, components = compute_visual_text_fusion_score(
            visual_description=visual_description,
            text_metadata=text_metadata,
            vo_segment=vo_segment,
            visual_text_weight=0.20,
            fusion_enabled=True,
        )

        assert score >= 0.0
        assert score <= 1.0
        assert "visual_only" in reason
        assert components['has_visual'] is True
        assert components['has_text'] is False

    def test_fusion_text_only(self):
        """Test fusion with only text signal available."""
        visual_description = ""  # Empty - no visual
        text_metadata = {
            "title": "Cooking Tutorial",
            "description": "How to make pasta",
            "tags": ["cooking", "pasta", "tutorial"]
        }
        vo_segment = "Learn how to cook pasta"

        score, reason, components = compute_visual_text_fusion_score(
            visual_description=visual_description,
            text_metadata=text_metadata,
            vo_segment=vo_segment,
            visual_text_weight=0.20,
            fusion_enabled=True,
        )

        assert score >= 0.0
        assert score <= 1.0
        assert "text_only" in reason
        assert components['has_visual'] is False
        assert components['has_text'] is True

    def test_fusion_no_signals(self):
        """Test fusion with no signals available."""
        visual_description = ""
        text_metadata = {}
        vo_segment = "Some voiceover text"

        score, reason, components = compute_visual_text_fusion_score(
            visual_description=visual_description,
            text_metadata=text_metadata,
            vo_segment=vo_segment,
            visual_text_weight=0.20,
            fusion_enabled=True,
        )

        assert score == 0.0
        assert "no_signals" in reason

    def test_fusion_disabled(self):
        """Test fusion when disabled."""
        visual_description = "Some visual content"
        text_metadata = {"title": "Test Title"}
        vo_segment = "Test voiceover"

        score, reason, components = compute_visual_text_fusion_score(
            visual_description=visual_description,
            text_metadata=text_metadata,
            vo_segment=vo_segment,
            visual_text_weight=0.20,
            fusion_enabled=False,
        )

        assert score == 0.0
        assert "fusion_disabled" in reason

    def test_custom_weight(self):
        """Test fusion with custom visual text weight."""
        visual_description = "A car driving on highway"
        text_metadata = {"title": "Highway Driving"}
        vo_segment = "Driving on the highway"

        score, reason, components = compute_visual_text_fusion_score(
            visual_description=visual_description,
            text_metadata=text_metadata,
            vo_segment=vo_segment,
            visual_text_weight=0.50,  # Higher visual weight
            fusion_enabled=True,
        )

        assert score >= 0.0
        assert score <= 1.0
        assert components['visual_text_weight'] == 0.50
        assert components['text_weight'] == 0.50

    def test_empty_vo_segment(self):
        """Test fusion with empty voiceover segment."""
        visual_description = "Some visual content"
        text_metadata = {"title": "Test Title"}
        vo_segment = ""

        score, reason, components = compute_visual_text_fusion_score(
            visual_description=visual_description,
            text_metadata=text_metadata,
            vo_segment=vo_segment,
            visual_text_weight=0.20,
            fusion_enabled=True,
        )

        assert score == 0.0


class TestComputeTextSimilarity:
    """Test _compute_text_similarity helper function."""

    def test_word_overlap_fallback(self):
        """Test word overlap similarity when no embedding provider."""
        text1 = "python programming tutorial"
        text2 = "learn python programming"

        score = _compute_text_similarity(text1, text2, embedding_provider=None)

        assert score >= 0.0
        assert score <= 1.0
        assert score > 0.0  # Should have some overlap

    def test_no_overlap(self):
        """Test with no word overlap."""
        text1 = "python programming"
        text2 = "cooking recipes"

        score = _compute_text_similarity(text1, text2, embedding_provider=None)

        assert score == 0.0

    def test_empty_texts(self):
        """Test with empty texts."""
        score = _compute_text_similarity("", "test", None)
        assert score == 0.0

        score = _compute_text_similarity("test", "", None)
        assert score == 0.0

        score = _compute_text_similarity("", "", None)
        assert score == 0.0
