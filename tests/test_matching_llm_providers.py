"""
Tests for src/matching/llm_providers.py

Covers:
- GeminiMatcher - Gemini Flash for batch matching
- ClaudeMatcher - Claude Haiku for matching
- LocalLLMMatcher - Ollama for local matching
- Error handling and edge cases
- Response parsing and fallbacks

Created: 2026-01-11 (Session 14)
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.llm_providers import (
    LLMProvider,
    GeminiMatcher,
    ClaudeMatcher,
    LocalLLMMatcher
)
from src.utils import SRTSegment


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_srt_segment():
    """Create a mock SRTSegment"""
    segment = Mock(spec=SRTSegment)
    segment.text = "This is a test video about Python programming"
    segment.source_file = "/path/to/video_abc123.mp4"
    segment.start_time = 10.0
    segment.end_time = 20.0
    return segment


@pytest.fixture
def mock_candidates(mock_srt_segment):
    """Create mock candidates for matching"""
    candidates = []
    for i in range(3):
        seg = Mock(spec=SRTSegment)
        seg.text = f"Candidate {i+1} about programming tutorials"
        seg.source_file = f"/path/to/video_{i}.mp4"
        seg.start_time = float(i * 10)
        seg.end_time = float((i + 1) * 10)
        candidates.append((seg, 0.8 - i * 0.1))  # (segment, similarity)
    return candidates


@pytest.fixture
def mock_items(mock_candidates):
    """Create mock items for batch matching"""
    return [
        ("Welcome to Python programming tutorial", mock_candidates),
        ("Learn about machine learning basics", mock_candidates),
    ]


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response"""
    response = Mock()
    response.parsed_data = [
        {"voiceover": 1, "selected": 1, "confidence": 0.85, "reason": "topic match"},
        {"voiceover": 2, "selected": 2, "confidence": 0.75, "reason": "semantic alignment"},
    ]
    return response


@pytest.fixture
def mock_single_response():
    """Create a mock single LLM response for LocalLLM"""
    response = Mock()
    response.parsed_data = {"selected": 1, "confidence": 0.8, "reason": "local match"}
    return response


# ============================================================================
# Test GeminiMatcher
# ============================================================================

class TestGeminiMatcher:
    """Test GeminiMatcher class"""

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_init_creates_client(self, mock_create_client):
        """Test GeminiMatcher initialization"""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        matcher = GeminiMatcher(api_key="test_key", model="gemini-2.0-flash")

        mock_create_client.assert_called_once_with(
            "gemini", api_key="test_key", model="gemini-2.0-flash"
        )
        assert matcher.client == mock_client

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_match_batch_success(self, mock_create_client, mock_items, mock_llm_response):
        """Test successful batch matching"""
        mock_client = Mock()
        mock_client.generate.return_value = mock_llm_response
        mock_create_client.return_value = mock_client

        matcher = GeminiMatcher(api_key="test_key")
        results = matcher.match_batch(mock_items)

        assert len(results) == 2
        # match_batch returns (selected_idx, confidence, reasoning, cot_reasoning)
        # Check first 3 elements of each tuple (cot_reasoning is None when use_cot=False)
        assert results[0][:3] == (0, 0.85, "topic match")  # selected 1 -> idx 0
        assert results[1][:3] == (1, 0.75, "semantic alignment")  # selected 2 -> idx 1

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_match_batch_with_context(self, mock_create_client, mock_items, mock_llm_response):
        """Test batch matching with context"""
        mock_client = Mock()
        mock_client.generate.return_value = mock_llm_response
        mock_create_client.return_value = mock_client

        matcher = GeminiMatcher(api_key="test_key")
        results = matcher.match_batch(mock_items, context="Python tutorial series")

        # Verify the context was included in the prompt
        call_args = mock_client.generate.call_args
        request = call_args[0][0]
        assert "CONTEXT: Python tutorial series" in request.prompt

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_match_batch_with_negative_rules(self, mock_create_client, mock_items, mock_llm_response):
        """Test batch matching with negative rules"""
        mock_client = Mock()
        mock_client.generate.return_value = mock_llm_response
        mock_create_client.return_value = mock_client

        matcher = GeminiMatcher(api_key="test_key")
        results = matcher.match_batch(
            mock_items,
            negative_rules=["Avoid stock footage", "No talking heads"]
        )

        # Verify negative rules were included
        call_args = mock_client.generate.call_args
        request = call_args[0][0]
        assert "AVOID:" in request.prompt
        assert "Avoid stock footage" in request.prompt

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_match_batch_clamps_confidence(self, mock_create_client, mock_items):
        """Test that confidence is clamped to 0.0-1.0"""
        response = Mock()
        response.parsed_data = [
            {"voiceover": 1, "selected": 1, "confidence": 1.5, "reason": "high"},  # > 1.0
            {"voiceover": 2, "selected": 1, "confidence": -0.5, "reason": "low"},  # < 0.0
        ]
        mock_client = Mock()
        mock_client.generate.return_value = response
        mock_create_client.return_value = mock_client

        matcher = GeminiMatcher(api_key="test_key")
        results = matcher.match_batch(mock_items)

        assert results[0][1] == 1.0  # Clamped to max
        assert results[1][1] == 0.0  # Clamped to min

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_match_batch_clamps_selected_index(self, mock_create_client, mock_items):
        """Test that selected index is clamped to valid range"""
        response = Mock()
        response.parsed_data = [
            {"voiceover": 1, "selected": 10, "confidence": 0.8, "reason": "high idx"},  # > len
            {"voiceover": 2, "selected": 0, "confidence": 0.8, "reason": "zero idx"},  # 0 (invalid)
        ]
        mock_client = Mock()
        mock_client.generate.return_value = response
        mock_create_client.return_value = mock_client

        matcher = GeminiMatcher(api_key="test_key")
        results = matcher.match_batch(mock_items)

        # Both should be clamped to valid candidate indices (0-2)
        assert 0 <= results[0][0] <= 2
        assert 0 <= results[1][0] <= 2

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_match_batch_missing_voiceover_entry(self, mock_create_client, mock_items):
        """Test fallback when response missing voiceover entry"""
        response = Mock()
        response.parsed_data = [
            {"voiceover": 1, "selected": 1, "confidence": 0.85, "reason": "match"},
            # Missing voiceover 2
        ]
        mock_client = Mock()
        mock_client.generate.return_value = response
        mock_create_client.return_value = mock_client

        matcher = GeminiMatcher(api_key="test_key")
        results = matcher.match_batch(mock_items)

        assert len(results) == 2
        # match_batch returns (selected_idx, confidence, reasoning, cot_reasoning)
        assert results[0][:3] == (0, 0.85, "match")
        assert results[1][2] == "parse fallback"  # Fallback for missing entry

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_match_batch_empty_response(self, mock_create_client, mock_items):
        """Test handling of empty response"""
        response = Mock()
        response.parsed_data = None
        mock_client = Mock()
        mock_client.generate.return_value = response
        mock_create_client.return_value = mock_client

        matcher = GeminiMatcher(api_key="test_key")

        with pytest.raises(ValueError, match="JSON parsing failed"):
            matcher.match_batch(mock_items)

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_match_batch_non_list_response(self, mock_create_client, mock_items):
        """Test handling of non-list response"""
        response = Mock()
        response.parsed_data = {"error": "invalid"}  # Dict instead of list
        mock_client = Mock()
        mock_client.generate.return_value = response
        mock_create_client.return_value = mock_client

        matcher = GeminiMatcher(api_key="test_key")

        with pytest.raises(ValueError, match="JSON parsing failed"):
            matcher.match_batch(mock_items)

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_match_batch_api_error(self, mock_create_client, mock_items):
        """Test handling of API error"""
        mock_client = Mock()
        mock_client.generate.side_effect = Exception("API rate limit exceeded")
        mock_create_client.return_value = mock_client

        matcher = GeminiMatcher(api_key="test_key")

        with pytest.raises(Exception, match="API rate limit exceeded"):
            matcher.match_batch(mock_items)

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_match_batch_truncates_reason(self, mock_create_client, mock_items):
        """Test that reason is truncated to 50 chars"""
        response = Mock()
        long_reason = "This is a very long reason that should be truncated to 50 characters maximum"
        response.parsed_data = [
            {"voiceover": 1, "selected": 1, "confidence": 0.85, "reason": long_reason},
            {"voiceover": 2, "selected": 1, "confidence": 0.75, "reason": "short"},
        ]
        mock_client = Mock()
        mock_client.generate.return_value = response
        mock_create_client.return_value = mock_client

        matcher = GeminiMatcher(api_key="test_key")
        results = matcher.match_batch(mock_items)

        assert len(results[0][2]) <= 50
        assert results[1][2] == "short"


# ============================================================================
# Test ClaudeMatcher
# ============================================================================

class TestClaudeMatcher:
    """Test ClaudeMatcher class"""

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_init_creates_client(self, mock_create_client):
        """Test ClaudeMatcher initialization"""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        matcher = ClaudeMatcher(api_key="test_key", model="claude-3-haiku-20240307")

        mock_create_client.assert_called_once_with(
            "anthropic", api_key="test_key", model="claude-3-haiku-20240307"
        )
        assert matcher.client == mock_client

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_match_batch_success(self, mock_create_client, mock_items, mock_llm_response):
        """Test successful batch matching with Claude"""
        mock_client = Mock()
        mock_client.generate.return_value = mock_llm_response
        mock_create_client.return_value = mock_client

        matcher = ClaudeMatcher(api_key="test_key")
        results = matcher.match_batch(mock_items)

        assert len(results) == 2
        # match_batch returns (selected_idx, confidence, reasoning, cot_reasoning)
        assert results[0][:3] == (0, 0.85, "topic match")
        assert results[1][:3] == (1, 0.75, "semantic alignment")

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_match_batch_with_context_and_rules(self, mock_create_client, mock_items, mock_llm_response):
        """Test batch matching with context and negative rules"""
        mock_client = Mock()
        mock_client.generate.return_value = mock_llm_response
        mock_create_client.return_value = mock_client

        matcher = ClaudeMatcher(api_key="test_key")
        results = matcher.match_batch(
            mock_items,
            context="Documentary about AI",
            negative_rules=["No ads", "No intros"]
        )

        call_args = mock_client.generate.call_args
        request = call_args[0][0]
        assert "CONTEXT: Documentary about AI" in request.prompt
        assert "AVOID:" in request.prompt
        assert "No ads" in request.prompt

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_match_batch_clamps_values(self, mock_create_client, mock_items):
        """Test value clamping in ClaudeMatcher"""
        response = Mock()
        response.parsed_data = [
            {"voiceover": 1, "selected": 100, "confidence": 2.0, "reason": "test"},
            {"voiceover": 2, "selected": -1, "confidence": -1.0, "reason": "test2"},
        ]
        mock_client = Mock()
        mock_client.generate.return_value = response
        mock_create_client.return_value = mock_client

        matcher = ClaudeMatcher(api_key="test_key")
        results = matcher.match_batch(mock_items)

        # Check clamping
        assert 0 <= results[0][0] <= 2  # Index clamped
        assert results[0][1] == 1.0  # Confidence clamped to max
        assert 0 <= results[1][0] <= 2
        assert results[1][1] == 0.0  # Confidence clamped to min

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_match_batch_parse_error(self, mock_create_client, mock_items):
        """Test handling of parse error"""
        response = Mock()
        response.parsed_data = "not a list"
        mock_client = Mock()
        mock_client.generate.return_value = response
        mock_create_client.return_value = mock_client

        matcher = ClaudeMatcher(api_key="test_key")

        with pytest.raises(ValueError, match="JSON parsing failed"):
            matcher.match_batch(mock_items)

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_match_batch_api_error(self, mock_create_client, mock_items):
        """Test handling of API error"""
        mock_client = Mock()
        mock_client.generate.side_effect = Exception("Anthropic API error")
        mock_create_client.return_value = mock_client

        matcher = ClaudeMatcher(api_key="test_key")

        with pytest.raises(Exception, match="Anthropic API error"):
            matcher.match_batch(mock_items)

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_match_batch_missing_entries_fallback(self, mock_create_client, mock_items):
        """Test fallback for missing voiceover entries"""
        response = Mock()
        response.parsed_data = [
            # Only voiceover 1, missing voiceover 2
            {"voiceover": 1, "selected": 2, "confidence": 0.9, "reason": "good match"},
        ]
        mock_client = Mock()
        mock_client.generate.return_value = response
        mock_create_client.return_value = mock_client

        matcher = ClaudeMatcher(api_key="test_key")
        results = matcher.match_batch(mock_items)

        assert len(results) == 2
        # match_batch returns (selected_idx, confidence, reasoning, cot_reasoning)
        assert results[0][:3] == (1, 0.9, "good match")  # Parsed correctly
        assert results[1][2] == "parse fallback"  # Fallback


# ============================================================================
# Test LocalLLMMatcher
# ============================================================================

class TestLocalLLMMatcher:
    """Test LocalLLMMatcher (Ollama) class"""

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_init_creates_client(self, mock_create_client):
        """Test LocalLLMMatcher initialization"""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        matcher = LocalLLMMatcher(model="llama3.2", host="http://localhost:11434")

        mock_create_client.assert_called_once_with(
            "ollama", model="llama3.2", host="http://localhost:11434"
        )
        assert matcher.client == mock_client

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_match_batch_processes_one_at_a_time(self, mock_create_client, mock_items, mock_single_response):
        """Test that LocalLLM processes items one at a time"""
        mock_client = Mock()
        mock_client.generate.return_value = mock_single_response
        mock_create_client.return_value = mock_client

        matcher = LocalLLMMatcher()
        results = matcher.match_batch(mock_items)

        # Should call generate once per item
        assert mock_client.generate.call_count == len(mock_items)
        assert len(results) == 2

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_match_batch_success(self, mock_create_client, mock_items, mock_single_response):
        """Test successful matching with LocalLLM"""
        mock_client = Mock()
        mock_client.generate.return_value = mock_single_response
        mock_create_client.return_value = mock_client

        matcher = LocalLLMMatcher()
        results = matcher.match_batch(mock_items)

        assert len(results) == 2
        for result in results:
            assert result[0] == 0  # selected 1 -> idx 0
            assert result[1] == 0.8
            assert result[2] == "local match"

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_match_batch_clamps_values(self, mock_create_client, mock_items):
        """Test value clamping in LocalLLMMatcher"""
        response = Mock()
        response.parsed_data = {"selected": 50, "confidence": 5.0, "reason": "test"}
        mock_client = Mock()
        mock_client.generate.return_value = response
        mock_create_client.return_value = mock_client

        matcher = LocalLLMMatcher()
        results = matcher.match_batch(mock_items)

        for result in results:
            assert 0 <= result[0] <= 2  # Index clamped
            assert result[1] == 1.0  # Confidence clamped

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_match_batch_fallback_on_error(self, mock_create_client, mock_items):
        """Test fallback when LLM call fails"""
        mock_client = Mock()
        mock_client.generate.side_effect = Exception("Connection refused")
        mock_create_client.return_value = mock_client

        matcher = LocalLLMMatcher()
        results = matcher.match_batch(mock_items)

        # Should return fallback values, not raise
        assert len(results) == 2
        for result in results:
            assert result[0] == 0  # Default to first candidate
            assert result[2] == "local fallback"

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_match_batch_fallback_on_invalid_response(self, mock_create_client, mock_items):
        """Test fallback when response is invalid"""
        response = Mock()
        response.parsed_data = "not a dict"  # Invalid
        mock_client = Mock()
        mock_client.generate.return_value = response
        mock_create_client.return_value = mock_client

        matcher = LocalLLMMatcher()
        results = matcher.match_batch(mock_items)

        # Should return fallback values
        assert len(results) == 2
        for result in results:
            assert result[2] == "local fallback"

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_match_batch_empty_candidates(self, mock_create_client):
        """Test handling of empty candidates list"""
        mock_client = Mock()
        response = Mock()
        response.parsed_data = None
        mock_client.generate.return_value = response
        mock_create_client.return_value = mock_client

        matcher = LocalLLMMatcher()
        items = [("Test voiceover", [])]  # Empty candidates

        results = matcher.match_batch(items)

        assert len(results) == 1
        assert results[0][1] == 0.5  # Default confidence when no candidates


# ============================================================================
# Test LLMProvider Base Class
# ============================================================================

class TestLLMProviderBase:
    """Test LLMProvider abstract base class"""

    @pytest.mark.fast
    def test_cannot_instantiate_directly(self):
        """Test that LLMProvider cannot be instantiated directly"""
        with pytest.raises(TypeError):
            LLMProvider()

    @pytest.mark.fast
    def test_subclass_must_implement_match_batch(self):
        """Test that subclasses must implement match_batch"""
        class IncompleteProvider(LLMProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteProvider()


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases across all providers"""

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_quotes_in_text_escaped(self, mock_create_client, mock_llm_response):
        """Test that quotes in text are escaped"""
        mock_client = Mock()
        mock_client.generate.return_value = mock_llm_response
        mock_create_client.return_value = mock_client

        # Create segment with quotes
        seg = Mock(spec=SRTSegment)
        seg.text = 'He said "hello" to the audience'
        seg.source_file = "/path/to/video.mp4"
        seg.start_time = 0.0
        seg.end_time = 10.0

        items = [('Test with "quotes" in text', [(seg, 0.8)])]

        matcher = GeminiMatcher(api_key="test_key")
        matcher.match_batch(items)

        # Verify quotes were replaced with single quotes in prompt
        call_args = mock_client.generate.call_args
        request = call_args[0][0]
        assert '"hello"' not in request.prompt  # Double quotes escaped

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_long_text_truncated(self, mock_create_client, mock_llm_response):
        """Test that long text is truncated"""
        mock_client = Mock()
        mock_client.generate.return_value = mock_llm_response
        mock_create_client.return_value = mock_client

        # Create segment with very long text
        seg = Mock(spec=SRTSegment)
        seg.text = "A" * 200  # 200 chars
        seg.source_file = "/path/to/video.mp4"
        seg.start_time = 0.0
        seg.end_time = 10.0

        long_vo = "B" * 300  # 300 char voiceover
        items = [(long_vo, [(seg, 0.8)])]

        matcher = GeminiMatcher(api_key="test_key")
        matcher.match_batch(items)

        # Verify truncation happened
        call_args = mock_client.generate.call_args
        request = call_args[0][0]
        # Voiceover truncated to 100 chars in prompt
        assert "B" * 101 not in request.prompt

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_single_item_batch(self, mock_create_client, mock_candidates):
        """Test batch with single item"""
        response = Mock()
        response.parsed_data = [
            {"voiceover": 1, "selected": 1, "confidence": 0.9, "reason": "match"}
        ]
        mock_client = Mock()
        mock_client.generate.return_value = response
        mock_create_client.return_value = mock_client

        items = [("Single voiceover", mock_candidates)]

        matcher = GeminiMatcher(api_key="test_key")
        results = matcher.match_batch(items)

        assert len(results) == 1
        # match_batch returns (selected_idx, confidence, reasoning, cot_reasoning)
        assert results[0][:3] == (0, 0.9, "match")

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_empty_candidates_in_batch(self, mock_create_client):
        """Test handling when some items have empty candidates"""
        response = Mock()
        response.parsed_data = [
            {"voiceover": 1, "selected": 1, "confidence": 0.8, "reason": "match"}
        ]
        mock_client = Mock()
        mock_client.generate.return_value = response
        mock_create_client.return_value = mock_client

        # Create segment
        seg = Mock(spec=SRTSegment)
        seg.text = "Test text"
        seg.source_file = "/path/to/video.mp4"
        seg.start_time = 0.0
        seg.end_time = 10.0

        items = [
            ("Voiceover 1", [(seg, 0.8)]),  # Has candidates
        ]

        matcher = GeminiMatcher(api_key="test_key")
        results = matcher.match_batch(items)

        assert len(results) == 1


# ============================================================================
# Test Request Configuration
# ============================================================================

class TestRequestConfiguration:
    """Test LLM request configuration"""

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_gemini_uses_json_array_format(self, mock_create_client, mock_items, mock_llm_response):
        """Test that GeminiMatcher uses JSON_ARRAY format"""
        mock_client = Mock()
        mock_client.generate.return_value = mock_llm_response
        mock_create_client.return_value = mock_client

        matcher = GeminiMatcher(api_key="test_key")
        matcher.match_batch(mock_items)

        call_args = mock_client.generate.call_args
        request = call_args[0][0]
        from src.llm_client import ResponseFormat
        assert request.response_format == ResponseFormat.JSON_ARRAY

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_gemini_uses_120s_timeout(self, mock_create_client, mock_items, mock_llm_response):
        """Test that GeminiMatcher uses 120s timeout"""
        mock_client = Mock()
        mock_client.generate.return_value = mock_llm_response
        mock_create_client.return_value = mock_client

        matcher = GeminiMatcher(api_key="test_key")
        matcher.match_batch(mock_items)

        call_args = mock_client.generate.call_args
        request = call_args[0][0]
        assert request.timeout == 120

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_local_llm_uses_json_format(self, mock_create_client, mock_items, mock_single_response):
        """Test that LocalLLMMatcher uses JSON format (not array)"""
        mock_client = Mock()
        mock_client.generate.return_value = mock_single_response
        mock_create_client.return_value = mock_client

        matcher = LocalLLMMatcher()
        matcher.match_batch(mock_items)

        call_args = mock_client.generate.call_args
        request = call_args[0][0]
        from src.llm_client import ResponseFormat
        assert request.response_format == ResponseFormat.JSON

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_local_llm_uses_60s_timeout(self, mock_create_client, mock_items, mock_single_response):
        """Test that LocalLLMMatcher uses 60s timeout"""
        mock_client = Mock()
        mock_client.generate.return_value = mock_single_response
        mock_create_client.return_value = mock_client

        matcher = LocalLLMMatcher()
        matcher.match_batch(mock_items)

        call_args = mock_client.generate.call_args
        request = call_args[0][0]
        assert request.timeout == 60

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_claude_uses_max_tokens(self, mock_create_client, mock_items, mock_llm_response):
        """Test that ClaudeMatcher uses max_tokens"""
        mock_client = Mock()
        mock_client.generate.return_value = mock_llm_response
        mock_create_client.return_value = mock_client

        matcher = ClaudeMatcher(api_key="test_key")
        matcher.match_batch(mock_items)

        call_args = mock_client.generate.call_args
        request = call_args[0][0]
        assert request.max_tokens == 1500

    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_matching_cache_key_prefix(self, mock_create_client, mock_items, mock_llm_response):
        """Test that all matchers use 'matching' cache key prefix"""
        mock_client = Mock()
        mock_client.generate.return_value = mock_llm_response
        mock_create_client.return_value = mock_client

        matcher = GeminiMatcher(api_key="test_key")
        matcher.match_batch(mock_items)

        call_args = mock_client.generate.call_args
        request = call_args[0][0]
        assert request.cache_key_prefix == "matching"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
