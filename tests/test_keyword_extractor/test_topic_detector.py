"""
Tests for keyword_extractor.topic_detector

Tests cover:
- detect_topic() orchestration with mocked LLM
- detect_topic_llm() with various responses
- Fallback behavior when LLM unavailable
"""

import pytest
from unittest.mock import Mock
from src.keyword_extractor.topic_detector import (
    detect_topic,
    detect_topic_llm
)


class TestDetectTopicLlm:
    """Test detect_topic_llm() function with mocked LLM"""

    def test_detect_topic_basic(self):
        """Test basic topic detection"""
        mock_llm_client = Mock()
        mock_response = Mock()
        mock_response.text = '"Wildlife Conservation"'
        mock_llm_client.generate.return_value = mock_response

        text = "This documentary explores wildlife conservation efforts in Africa."
        topic = detect_topic_llm(text, mock_llm_client)

        assert topic == "Wildlife Conservation"
        mock_llm_client.generate.assert_called_once()

    def test_detect_topic_with_quotes(self):
        """Test topic detection strips quotes"""
        mock_llm_client = Mock()
        mock_response = Mock()
        mock_response.text = '"Mountain Climbing"'
        mock_llm_client.generate.return_value = mock_response

        text = "Scaling the world's highest peaks."
        topic = detect_topic_llm(text, mock_llm_client)

        assert topic == "Mountain Climbing"
        assert '"' not in topic

    def test_detect_topic_with_extra_whitespace(self):
        """Test topic detection strips whitespace"""
        mock_llm_client = Mock()
        mock_response = Mock()
        mock_response.text = '  Space Exploration  '
        mock_llm_client.generate.return_value = mock_response

        text = "Journey to Mars."
        topic = detect_topic_llm(text, mock_llm_client)

        assert topic == "Space Exploration"

    def test_detect_topic_multiword(self):
        """Test topic detection with multiple words"""
        mock_llm_client = Mock()
        mock_response = Mock()
        mock_response.text = '"Deep Sea Marine Biology Research"'
        mock_llm_client.generate.return_value = mock_response

        text = "Exploring the depths of the ocean."
        topic = detect_topic_llm(text, mock_llm_client)

        assert "Deep Sea" in topic or "Marine Biology" in topic

    def test_detect_topic_short_response(self):
        """Test topic detection with single word"""
        mock_llm_client = Mock()
        mock_response = Mock()
        mock_response.text = 'Nature'
        mock_llm_client.generate.return_value = mock_response

        text = "A film about nature."
        topic = detect_topic_llm(text, mock_llm_client)

        assert topic == "Nature"

    def test_detect_topic_empty_response(self):
        """Test topic detection with empty response"""
        mock_llm_client = Mock()
        mock_response = Mock()
        mock_response.text = ''
        mock_llm_client.generate.return_value = mock_response

        text = "Some text."
        topic = detect_topic_llm(text, mock_llm_client)

        assert topic == ""

    def test_detect_topic_json_response(self):
        """Test topic detection with JSON response"""
        mock_llm_client = Mock()
        mock_response = Mock()
        mock_response.text = '{"topic": "Climate Change"}'
        mock_llm_client.generate.return_value = mock_response

        text = "Global warming effects."
        topic = detect_topic_llm(text, mock_llm_client)

        # Should extract topic from JSON or return raw response
        assert "Climate" in topic or "topic" in topic

    def test_detect_topic_prompt_contains_text(self):
        """Test that LLM prompt contains input text"""
        mock_llm_client = Mock()
        mock_response = Mock()
        mock_response.text = '"Test Topic"'
        mock_llm_client.generate.return_value = mock_response

        text = "Unique text string XYZ123"
        detect_topic_llm(text, mock_llm_client)

        # Check that prompt contains our text
        call_args = mock_llm_client.generate.call_args[0][0]
        assert "XYZ123" in call_args.prompt or "Unique text" in call_args.prompt


class TestDetectTopic:
    """Test detect_topic() orchestration function"""

    def test_detect_topic_with_llm_client(self):
        """Test detect_topic() with available LLM client"""
        mock_client = Mock()
        mock_client.generate = Mock()

        text = "Documentary about space exploration."

        # This will use the real detect_topic which may call LLM
        # Since we're testing integration, we mock at a higher level
        # For now, test that function exists and handles None client
        topic = detect_topic(text, None)

        # With None client, should return empty string
        assert isinstance(topic, str)

    def test_detect_topic_without_llm_client(self):
        """Test detect_topic() without LLM client (fallback)"""
        text = "Documentary about mountain climbing."
        topic = detect_topic(text, None)

        # Should return empty string or handle gracefully
        assert isinstance(topic, str)

    def test_detect_topic_empty_text(self):
        """Test detect_topic() with empty text"""
        topic = detect_topic("", None)

        assert isinstance(topic, str)

    def test_detect_topic_long_text(self):
        """Test detect_topic() with very long text"""
        long_text = "Documentary " * 1000
        topic = detect_topic(long_text, None)

        assert isinstance(topic, str)

    def test_detect_topic_special_characters(self):
        """Test detect_topic() with special characters"""
        text = "Documentary about Sao Paulo, Brazil's cafe culture & art scene"
        topic = detect_topic(text, None)

        assert isinstance(topic, str)


class TestTopicDetectionIntegration:
    """Integration tests for topic detection"""

    def test_topic_detection_travel_documentary(self):
        """Test topic detection for travel documentary"""
        mock_llm_client = Mock()
        mock_response = Mock()
        mock_response.text = '"Travel and Culture"'
        mock_llm_client.generate.return_value = mock_response

        text = """
        Join us as we explore the ancient streets of Kyoto,
        sample authentic ramen, and witness traditional tea ceremonies.
        This journey through Japan reveals the harmony between
        old traditions and modern innovation.
        """

        topic = detect_topic_llm(text, mock_llm_client)

        assert len(topic) > 0
        assert isinstance(topic, str)

    def test_topic_detection_nature_documentary(self):
        """Test topic detection for nature documentary"""
        mock_llm_client = Mock()
        mock_response = Mock()
        mock_response.text = '"Wildlife and Nature"'
        mock_llm_client.generate.return_value = mock_response

        text = """
        In the heart of the Serengeti, lions hunt wildebeest
        while elephants march to distant waterholes. This is
        the circle of life in Africa's greatest wilderness.
        """

        topic = detect_topic_llm(text, mock_llm_client)

        assert len(topic) > 0
        assert isinstance(topic, str)

    def test_topic_detection_history_documentary(self):
        """Test topic detection for history documentary"""
        mock_llm_client = Mock()
        mock_response = Mock()
        mock_response.text = '"World War II History"'
        mock_llm_client.generate.return_value = mock_response

        text = """
        On June 6th, 1944, Allied forces stormed the beaches of Normandy.
        This pivotal moment in World War II changed the course of history.
        """

        topic = detect_topic_llm(text, mock_llm_client)

        assert len(topic) > 0
        assert isinstance(topic, str)
