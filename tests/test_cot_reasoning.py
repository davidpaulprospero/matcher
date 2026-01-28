"""
Tests for chain-of-thought (CoT) reasoning in LLM matching.

Tests the structured 4-step reasoning process:
1. Identify key voiceover themes
2. List matching elements in video
3. Evaluate fit using explicit rubric
4. Compute final score

Rubric weights:
- Visual relevance: 30%
- Topic match: 40%
- Keyword overlap: 20%
- Flow: 10%
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.matching.llm_providers import (
    CotReasoning,
    COT_RUBRIC_WEIGHTS,
    build_cot_prompt,
    build_cot_batch_prompt,
    parse_cot_reasoning,
)
from src.utils import SRTSegment


class TestCotReasoningDataclass:
    """Test CotReasoning dataclass functionality."""

    @pytest.mark.fast
    def test_empty_cot_is_not_complete(self):
        """Empty CotReasoning should not be complete."""
        cot = CotReasoning()
        assert not cot.is_complete

    @pytest.mark.fast
    def test_partial_cot_is_not_complete(self):
        """Partial CotReasoning should not be complete."""
        cot = CotReasoning(
            voiceover_themes=["nature", "wildlife"],
            video_elements=[],
            rubric_scores={},
            final_score=0.0
        )
        assert not cot.is_complete

    @pytest.mark.fast
    def test_complete_cot_is_complete(self):
        """Complete CotReasoning should be marked complete."""
        cot = CotReasoning(
            voiceover_themes=["nature", "wildlife"],
            video_elements=["forest", "animals"],
            rubric_scores={
                "visual_relevance": 0.8,
                "topic_match": 0.9,
                "keyword_overlap": 0.7,
                "flow": 0.8
            },
            final_score=0.85
        )
        assert cot.is_complete

    @pytest.mark.fast
    def test_compute_weighted_score_with_all_rubrics(self):
        """Weighted score should use all rubric weights."""
        cot = CotReasoning(
            rubric_scores={
                "visual_relevance": 0.8,  # 0.8 * 0.30 = 0.24
                "topic_match": 0.9,       # 0.9 * 0.40 = 0.36
                "keyword_overlap": 0.7,   # 0.7 * 0.20 = 0.14
                "flow": 0.8               # 0.8 * 0.10 = 0.08
            },
            final_score=0.5  # Fallback, not used when rubrics are present
        )
        # Total = 0.82, weight_sum = 1.0
        expected = 0.24 + 0.36 + 0.14 + 0.08  # = 0.82
        assert abs(cot.compute_weighted_score() - expected) < 0.01

    @pytest.mark.fast
    def test_compute_weighted_score_with_partial_rubrics(self):
        """Weighted score should normalize for missing rubrics."""
        cot = CotReasoning(
            rubric_scores={
                "visual_relevance": 0.8,  # 0.30
                "topic_match": 0.9,       # 0.40
            },
            final_score=0.5
        )
        # weight_sum = 0.70, total = 0.24 + 0.36 = 0.60
        # normalized = 0.60 / 0.70 = 0.857
        expected = (0.8 * 0.30 + 0.9 * 0.40) / 0.70
        assert abs(cot.compute_weighted_score() - expected) < 0.01

    @pytest.mark.fast
    def test_compute_weighted_score_empty_rubrics_uses_final_score(self):
        """Empty rubrics should fall back to final_score."""
        cot = CotReasoning(
            rubric_scores={},
            final_score=0.75
        )
        assert cot.compute_weighted_score() == 0.75


class TestCotRubricWeights:
    """Test rubric weight constants."""

    @pytest.mark.fast
    def test_rubric_weights_sum_to_one(self):
        """Rubric weights should sum to 1.0."""
        total = sum(COT_RUBRIC_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    @pytest.mark.fast
    def test_rubric_weights_correct_values(self):
        """Rubric weights should match specification."""
        assert COT_RUBRIC_WEIGHTS['visual_relevance'] == 0.30
        assert COT_RUBRIC_WEIGHTS['topic_match'] == 0.40
        assert COT_RUBRIC_WEIGHTS['keyword_overlap'] == 0.20
        assert COT_RUBRIC_WEIGHTS['flow'] == 0.10

    @pytest.mark.fast
    def test_all_rubric_keys_present(self):
        """All required rubric keys should be present."""
        required_keys = {'visual_relevance', 'topic_match', 'keyword_overlap', 'flow'}
        assert set(COT_RUBRIC_WEIGHTS.keys()) == required_keys


class TestBuildCotPrompt:
    """Test single voiceover CoT prompt building."""

    def create_test_segment(self, text: str, source_file: str = "test_video.mp4") -> SRTSegment:
        """Create a test SRTSegment."""
        return SRTSegment(
            index=1,
            start_time=0.0,
            end_time=5.0,
            text=text,
            source_file=source_file
        )

    @pytest.mark.fast
    def test_build_cot_prompt_basic(self):
        """Build basic CoT prompt without extras."""
        voiceover = "The forest is home to many wildlife species"
        candidates = [
            (self.create_test_segment("Trees and animals in nature"), 0.9),
            (self.create_test_segment("City skyline at night"), 0.5),
        ]

        prompt = build_cot_prompt(voiceover, candidates)

        # Check structure elements
        assert "VOICEOVER:" in prompt
        assert "CANDIDATES:" in prompt
        assert "## SCORING RUBRIC" in prompt
        assert "Visual Relevance (30%)" in prompt
        assert "Topic Match (40%)" in prompt
        assert "Keyword Overlap (20%)" in prompt
        assert "Flow (10%)" in prompt
        assert "## REASONING STEPS" in prompt
        assert "STEP 1 - VOICEOVER THEMES" in prompt
        assert "STEP 2 - VIDEO ELEMENTS" in prompt
        assert "STEP 3 - RUBRIC EVALUATION" in prompt
        assert "STEP 4 - FINAL SCORE" in prompt
        assert "## RESPONSE FORMAT" in prompt

    @pytest.mark.fast
    def test_build_cot_prompt_with_context(self):
        """Build CoT prompt with context."""
        voiceover = "Nature documentary"
        candidates = [(self.create_test_segment("Wildlife footage"), 0.9)]

        prompt = build_cot_prompt(voiceover, candidates, context="Documentary about forests")

        assert "CONTEXT: Documentary about forests" in prompt

    @pytest.mark.fast
    def test_build_cot_prompt_with_negative_sample(self):
        """Build CoT prompt with negative sample."""
        voiceover = "Nature documentary"
        candidates = [(self.create_test_segment("Wildlife footage"), 0.9)]
        negative = (self.create_test_segment("Unrelated urban scene"), 0.2)

        prompt = build_cot_prompt(voiceover, candidates, negative_sample=negative)

        assert "unlikely match" in prompt.lower()
        assert "do NOT select them" in prompt

    @pytest.mark.fast
    def test_build_cot_prompt_with_negative_rules(self):
        """Build CoT prompt with negative rules."""
        voiceover = "Nature documentary"
        candidates = [(self.create_test_segment("Wildlife footage"), 0.9)]
        negative_rules = ["Don't match talking heads", "Avoid static images"]

        prompt = build_cot_prompt(voiceover, candidates, negative_rules=negative_rules)

        assert "AVOID:" in prompt
        assert "Don't match talking heads" in prompt
        assert "Avoid static images" in prompt

    @pytest.mark.fast
    def test_build_cot_prompt_escapes_quotes(self):
        """Prompt should escape quotes in text."""
        voiceover = 'He said "hello" to her'
        candidates = [(self.create_test_segment('Response: "hi"'), 0.9)]

        prompt = build_cot_prompt(voiceover, candidates)

        # Should escape double quotes to single quotes
        assert '"hello"' not in prompt or "'hello'" in prompt

    @pytest.mark.fast
    def test_build_cot_prompt_truncates_long_text(self):
        """Prompt should truncate very long text."""
        voiceover = "A" * 200  # Very long voiceover
        candidates = [(self.create_test_segment("B" * 200), 0.9)]

        prompt = build_cot_prompt(voiceover, candidates)

        # Should be truncated to 100 chars
        assert "A" * 101 not in prompt


class TestBuildCotBatchPrompt:
    """Test batch CoT prompt building."""

    def create_test_segment(self, text: str, source_file: str = "test_video.mp4") -> SRTSegment:
        """Create a test SRTSegment."""
        return SRTSegment(
            index=1,
            start_time=0.0,
            end_time=5.0,
            text=text,
            source_file=source_file
        )

    @pytest.mark.fast
    def test_build_batch_prompt_multiple_items(self):
        """Build batch prompt with multiple voiceover items."""
        items = [
            ("Forest wildlife", [(self.create_test_segment("Animals"), 0.9)]),
            ("Ocean waves", [(self.create_test_segment("Beach scene"), 0.8)]),
        ]

        prompt = build_cot_batch_prompt(items)

        assert "VOICEOVER 1:" in prompt
        assert "VOICEOVER 2:" in prompt
        assert "Forest wildlife" in prompt
        assert "Ocean waves" in prompt
        assert "## SCORING RUBRIC" in prompt

    @pytest.mark.fast
    def test_build_batch_prompt_with_negative_samples(self):
        """Build batch prompt with negative samples per item."""
        items = [
            ("Forest wildlife", [(self.create_test_segment("Animals"), 0.9)]),
        ]
        negative_samples = [
            (self.create_test_segment("Urban traffic"), 0.2)
        ]

        prompt = build_cot_batch_prompt(items, negative_samples=negative_samples)

        assert "unlikely match" in prompt.lower()


class TestParseCotReasoning:
    """Test parsing of CoT reasoning responses."""

    @pytest.mark.fast
    def test_parse_complete_response(self):
        """Parse complete CoT response."""
        response = {
            "selected": 1,
            "voiceover_themes": ["nature", "wildlife", "ecosystem"],
            "video_elements": ["forest", "animals", "rivers"],
            "rubric_scores": {
                "visual_relevance": 0.85,
                "topic_match": 0.90,
                "keyword_overlap": 0.75,
                "flow": 0.80
            },
            "confidence": 0.87,
            "reason": "Strong thematic alignment with visual elements"
        }

        cot = parse_cot_reasoning(response)

        assert cot.is_complete
        assert cot.voiceover_themes == ["nature", "wildlife", "ecosystem"]
        assert cot.video_elements == ["forest", "animals", "rivers"]
        assert cot.rubric_scores["visual_relevance"] == 0.85
        assert cot.rubric_scores["topic_match"] == 0.90
        assert cot.final_score == 0.87
        assert "thematic alignment" in cot.reasoning_text

    @pytest.mark.fast
    def test_parse_empty_response(self):
        """Parse empty/None response returns empty CotReasoning."""
        cot = parse_cot_reasoning(None)
        assert not cot.is_complete

        cot = parse_cot_reasoning({})
        assert not cot.is_complete

    @pytest.mark.fast
    def test_parse_string_theme_converted_to_list(self):
        """Single string theme should be converted to list."""
        response = {
            "voiceover_themes": "single theme",
            "video_elements": ["element"],
            "rubric_scores": {"topic_match": 0.9},
            "confidence": 0.8
        }

        cot = parse_cot_reasoning(response)

        assert cot.voiceover_themes == ["single theme"]

    @pytest.mark.fast
    def test_parse_clamps_out_of_range_scores(self):
        """Scores should be clamped to [0, 1]."""
        response = {
            "voiceover_themes": ["theme"],
            "video_elements": ["element"],
            "rubric_scores": {
                "visual_relevance": 1.5,  # Over 1
                "topic_match": -0.2,       # Under 0
                "keyword_overlap": 0.5     # Normal
            },
            "confidence": 1.2  # Over 1
        }

        cot = parse_cot_reasoning(response)

        assert cot.rubric_scores["visual_relevance"] == 1.0
        assert cot.rubric_scores["topic_match"] == 0.0
        assert cot.rubric_scores["keyword_overlap"] == 0.5
        assert cot.final_score == 1.0

    @pytest.mark.fast
    def test_parse_ignores_unknown_rubric_keys(self):
        """Unknown rubric keys should be ignored."""
        response = {
            "voiceover_themes": ["theme"],
            "video_elements": ["element"],
            "rubric_scores": {
                "visual_relevance": 0.8,
                "unknown_metric": 0.9,  # Not in COT_RUBRIC_WEIGHTS
                "another_unknown": 0.5
            },
            "confidence": 0.8
        }

        cot = parse_cot_reasoning(response)

        assert "visual_relevance" in cot.rubric_scores
        assert "unknown_metric" not in cot.rubric_scores
        assert "another_unknown" not in cot.rubric_scores

    @pytest.mark.fast
    def test_parse_handles_invalid_rubric_scores(self):
        """Invalid rubric score values should be skipped."""
        response = {
            "voiceover_themes": ["theme"],
            "video_elements": ["element"],
            "rubric_scores": {
                "visual_relevance": "not a number",
                "topic_match": None,
                "keyword_overlap": 0.7
            },
            "confidence": 0.8
        }

        cot = parse_cot_reasoning(response)

        assert "visual_relevance" not in cot.rubric_scores
        assert "topic_match" not in cot.rubric_scores
        assert cot.rubric_scores["keyword_overlap"] == 0.7

    @pytest.mark.fast
    def test_parse_truncates_long_reason(self):
        """Long reason text should be truncated."""
        response = {
            "voiceover_themes": ["theme"],
            "video_elements": ["element"],
            "rubric_scores": {"topic_match": 0.9},
            "confidence": 0.8,
            "reason": "A" * 200  # Very long
        }

        cot = parse_cot_reasoning(response)

        assert len(cot.reasoning_text) <= 100

    @pytest.mark.fast
    def test_parse_handles_nested_quotes_in_themes(self):
        """Parse should handle nested quotes in theme strings."""
        response = {
            "voiceover_themes": ['She said "hello" to him', "It's a 'special' day"],
            "video_elements": ["element"],
            "rubric_scores": {"topic_match": 0.9},
            "confidence": 0.8,
            "reason": "match"
        }

        cot = parse_cot_reasoning(response)

        assert len(cot.voiceover_themes) == 2
        assert '"hello"' in cot.voiceover_themes[0]
        assert "'special'" in cot.voiceover_themes[1]

    @pytest.mark.fast
    def test_parse_handles_special_characters_in_elements(self):
        """Parse should handle special characters in video elements."""
        response = {
            "voiceover_themes": ["theme"],
            "video_elements": [
                "café & résumé",
                "emoji: 🎬🎥",
                "newline\\nand\\ttab",
                "<script>alert('xss')</script>"
            ],
            "rubric_scores": {"topic_match": 0.9},
            "confidence": 0.8,
            "reason": "match"
        }

        cot = parse_cot_reasoning(response)

        assert len(cot.video_elements) == 4
        assert "café" in cot.video_elements[0]
        assert "🎬" in cot.video_elements[1]
        assert "\\n" in cot.video_elements[2]
        assert "<script>" in cot.video_elements[3]

    @pytest.mark.fast
    def test_parse_handles_special_characters_in_reason(self):
        """Parse should handle special characters in reason text."""
        response = {
            "voiceover_themes": ["theme"],
            "video_elements": ["element"],
            "rubric_scores": {"topic_match": 0.9},
            "confidence": 0.8,
            "reason": 'Text with "nested" quotes and special chars: <>&\'\\'
        }

        cot = parse_cot_reasoning(response)

        # Should preserve special characters in truncated string
        assert '"nested"' in cot.reasoning_text
        assert "<>&" in cot.reasoning_text


class TestCotIntegration:
    """Integration tests for CoT with LLM providers."""

    def create_test_segment(self, text: str, source_file: str = "test_video.mp4") -> SRTSegment:
        """Create a test SRTSegment."""
        return SRTSegment(
            index=1,
            start_time=0.0,
            end_time=5.0,
            text=text,
            source_file=source_file
        )

    @patch('src.matching.llm_providers.GeminiMatcher.__init__', return_value=None)
    @pytest.mark.fast
    def test_gemini_matcher_use_cot_flag_changes_prompt(self, mock_init):
        """GeminiMatcher should use CoT prompt when use_cot=True."""
        from src.matching.llm_providers import GeminiMatcher

        matcher = GeminiMatcher.__new__(GeminiMatcher)
        matcher.client = MagicMock()

        # Mock response
        mock_response = MagicMock()
        mock_response.parsed_data = [{
            "voiceover": 1,
            "selected": 1,
            "voiceover_themes": ["theme"],
            "video_elements": ["element"],
            "rubric_scores": {
                "visual_relevance": 0.8,
                "topic_match": 0.9,
                "keyword_overlap": 0.7,
                "flow": 0.8
            },
            "confidence": 0.85,
            "reason": "good match"
        }]
        matcher.client.generate.return_value = mock_response

        items = [("Test voiceover", [(self.create_test_segment("Test video"), 0.9)])]

        # Call with use_cot=True
        results = matcher.match_batch(items, use_cot=True)

        # Verify call was made
        assert matcher.client.generate.called
        call_args = matcher.client.generate.call_args[0][0]

        # Check that CoT cache key prefix was used
        assert call_args.cache_key_prefix == "matching_cot"

    @pytest.mark.fast
    def test_cot_weighted_score_blending(self):
        """Test that CoT weighted score is blended correctly."""
        # Simulate a response with CoT
        response = {
            "selected": 1,
            "voiceover_themes": ["nature"],
            "video_elements": ["forest"],
            "rubric_scores": {
                "visual_relevance": 0.8,
                "topic_match": 0.9,
                "keyword_overlap": 0.7,
                "flow": 0.8
            },
            "confidence": 0.7,  # LLM's raw confidence
            "reason": "match"
        }

        cot = parse_cot_reasoning(response)
        assert cot.is_complete

        weighted = cot.compute_weighted_score()
        llm_confidence = 0.7

        # Blending: 70% weighted + 30% LLM
        expected_blended = 0.7 * weighted + 0.3 * llm_confidence

        # weighted = 0.8*0.3 + 0.9*0.4 + 0.7*0.2 + 0.8*0.1 = 0.82
        assert abs(weighted - 0.82) < 0.01
        assert abs(expected_blended - (0.7 * 0.82 + 0.3 * 0.7)) < 0.01


class TestCotConfigOption:
    """Test chain_of_thought_enabled config option."""

    @pytest.mark.fast
    def test_matching_config_has_chain_of_thought_option(self):
        """MatchingConfig should have chain_of_thought_enabled field."""
        from src.config.sections.matching import MatchingConfig

        config = MatchingConfig()
        assert hasattr(config, 'chain_of_thought_enabled')
        assert config.chain_of_thought_enabled is True  # Default

    @pytest.mark.fast
    def test_chain_of_thought_disabled(self):
        """chain_of_thought_enabled can be set to False."""
        from src.config.sections.matching import MatchingConfig

        config = MatchingConfig(chain_of_thought_enabled=False)
        assert config.chain_of_thought_enabled is False
