"""
Tests for keyword_extractor.prompts

Tests cover:
- All 6 prompts exist and are non-empty strings
- Prompts contain required placeholder variables
- Prompts compile with sample data
"""

import pytest
from src.keyword_extractor import prompts


class TestPromptExistence:
    """Test that all prompts exist and are valid"""

    def test_all_prompts_exist(self):
        """Test all 6 expected prompts exist"""
        required_prompts = [
            'SEGMENT_KEYWORD_PROMPT',
            'BATCH_SEGMENT_KEYWORDS_PROMPT',
            'KEYWORD_EXTRACTION_PROMPT',
            'ENTITY_EXTRACTION_PROMPT',
            'KEYWORD_EXPANSION_PROMPT',
            'TOPIC_DETECTION_PROMPT'
        ]
        for prompt_name in required_prompts:
            assert hasattr(prompts, prompt_name), f"Missing prompt: {prompt_name}"

    def test_prompts_are_strings(self):
        """Test all prompts are strings"""
        all_prompts = [
            prompts.SEGMENT_KEYWORD_PROMPT,
            prompts.BATCH_SEGMENT_KEYWORDS_PROMPT,
            prompts.KEYWORD_EXTRACTION_PROMPT,
            prompts.ENTITY_EXTRACTION_PROMPT,
            prompts.KEYWORD_EXPANSION_PROMPT,
            prompts.TOPIC_DETECTION_PROMPT
        ]
        for prompt in all_prompts:
            assert isinstance(prompt, str)
            assert len(prompt) > 0


class TestPromptPlaceholders:
    """Test that prompts have required placeholders"""

    def test_segment_keyword_prompt_placeholders(self):
        """Test SEGMENT_KEYWORD_PROMPT has required placeholders"""
        prompt = prompts.SEGMENT_KEYWORD_PROMPT
        assert '{segment_text}' in prompt
        assert '{topic}' in prompt

    def test_batch_segment_keywords_prompt_placeholders(self):
        """Test BATCH_SEGMENT_KEYWORDS_PROMPT has required placeholders"""
        prompt = prompts.BATCH_SEGMENT_KEYWORDS_PROMPT
        assert '{segments_text}' in prompt
        assert '{topic}' in prompt

    def test_keyword_extraction_prompt_placeholders(self):
        """Test KEYWORD_EXTRACTION_PROMPT has required placeholders"""
        prompt = prompts.KEYWORD_EXTRACTION_PROMPT
        assert '{voiceover_text}' in prompt
        assert '{max_keywords}' in prompt

    def test_entity_extraction_prompt_placeholders(self):
        """Test ENTITY_EXTRACTION_PROMPT has required placeholders"""
        prompt = prompts.ENTITY_EXTRACTION_PROMPT
        assert '{text}' in prompt
        # Note: topic is NOT a placeholder in this prompt - entities extracted from text only

    def test_keyword_expansion_prompt_placeholders(self):
        """Test KEYWORD_EXPANSION_PROMPT has required placeholders"""
        prompt = prompts.KEYWORD_EXPANSION_PROMPT
        assert '{initial_keywords}' in prompt
        assert '{topic}' in prompt
        assert '{max_keywords}' in prompt

    def test_topic_detection_prompt_placeholders(self):
        """Test TOPIC_DETECTION_PROMPT has required placeholders"""
        prompt = prompts.TOPIC_DETECTION_PROMPT
        assert '{text}' in prompt


class TestPromptFormatting:
    """Test that prompts can be formatted with sample data"""

    def test_segment_keyword_prompt_formatting(self):
        """Test SEGMENT_KEYWORD_PROMPT formats correctly"""
        prompt = prompts.SEGMENT_KEYWORD_PROMPT.format(
            segment_text="Mountains are tall.",
            topic="Geography"
        )
        assert "Mountains are tall." in prompt
        assert "Geography" in prompt
        assert '{' not in prompt  # No unfilled placeholders

    def test_batch_segment_keywords_prompt_formatting(self):
        """Test BATCH_SEGMENT_KEYWORDS_PROMPT formats correctly"""
        prompt = prompts.BATCH_SEGMENT_KEYWORDS_PROMPT.format(
            segments_text="Segment 1\nSegment 2",
            topic="Nature"
        )
        assert "Segment 1" in prompt
        assert "Nature" in prompt
        assert '{' not in prompt

    def test_keyword_extraction_prompt_formatting(self):
        """Test KEYWORD_EXTRACTION_PROMPT formats correctly"""
        prompt = prompts.KEYWORD_EXTRACTION_PROMPT.format(
            voiceover_text="Documentary about mountains.",
            max_keywords=30
        )
        assert "Documentary about mountains." in prompt
        assert "30" in prompt
        assert '{' not in prompt

    def test_entity_extraction_prompt_formatting(self):
        """Test ENTITY_EXTRACTION_PROMPT formats correctly"""
        prompt = prompts.ENTITY_EXTRACTION_PROMPT.format(
            text="Mount Everest is in Nepal."
        )
        assert "Mount Everest is in Nepal." in prompt
        # Check no unfilled placeholders (prompt contains JSON examples with {{ }})
        assert '{text}' not in prompt

    def test_keyword_expansion_prompt_formatting(self):
        """Test KEYWORD_EXPANSION_PROMPT formats correctly"""
        prompt = prompts.KEYWORD_EXPANSION_PROMPT.format(
            initial_keywords=["mountain", "snow"],
            topic="Nature",
            max_keywords=50
        )
        assert "mountain" in prompt
        assert "Nature" in prompt
        assert "50" in prompt
        assert '{' not in prompt

    def test_topic_detection_prompt_formatting(self):
        """Test TOPIC_DETECTION_PROMPT formats correctly"""
        prompt = prompts.TOPIC_DETECTION_PROMPT.format(
            text="This is a documentary about wildlife."
        )
        assert "This is a documentary about wildlife." in prompt
        assert '{' not in prompt


class TestPromptContent:
    """Test prompt content has expected instructions"""

    def test_batch_segment_keywords_prompt_has_json_instruction(self):
        """Test BATCH_SEGMENT_KEYWORDS_PROMPT asks for JSON output"""
        prompt = prompts.BATCH_SEGMENT_KEYWORDS_PROMPT.lower()
        assert 'json' in prompt

    def test_keyword_extraction_prompt_has_youtube_mention(self):
        """Test KEYWORD_EXTRACTION_PROMPT mentions YouTube search"""
        prompt = prompts.KEYWORD_EXTRACTION_PROMPT.lower()
        assert 'youtube' in prompt or 'search' in prompt

    def test_entity_extraction_prompt_has_entity_types(self):
        """Test ENTITY_EXTRACTION_PROMPT mentions entity types"""
        prompt = prompts.ENTITY_EXTRACTION_PROMPT.lower()
        # Should mention at least some entity types
        assert any(term in prompt for term in ['person', 'place', 'location', 'organization', 'date'])

    def test_prompts_have_reasonable_length(self):
        """Test prompts are not too short or too long"""
        all_prompts = [
            prompts.SEGMENT_KEYWORD_PROMPT,
            prompts.BATCH_SEGMENT_KEYWORDS_PROMPT,
            prompts.KEYWORD_EXTRACTION_PROMPT,
            prompts.ENTITY_EXTRACTION_PROMPT,
            prompts.KEYWORD_EXPANSION_PROMPT,
            prompts.TOPIC_DETECTION_PROMPT
        ]
        for prompt in all_prompts:
            assert len(prompt) > 50, "Prompt too short"
            assert len(prompt) < 5000, "Prompt too long"
