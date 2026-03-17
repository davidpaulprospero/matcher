"""
Unit tests for GrokClient research client.

Tests cover:
- US-003: Grok research client unit tests
  - AC1: Test GrokClient.analyze() returns correctly parsed AnalysisResult objects
  - AC2: Test GrokClient.analyze() handles connection errors with retry logic
  - AC3: Test GrokClient.analyze() handles empty response gracefully
  - AC4: Test GrokClient.analyze() respects configured timeout settings
  - AC5: Test GrokClient validates input parameters before making requests

Note: GrokClient uses research() method (not analyze()) per implementation.
      The API uses OpenAI-compatible chat/completions endpoint.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import requests


# ---------------------------------------------------------------------------
# AC1: Test GrokClient.research() returns correctly parsed ResearchResult objects
# ---------------------------------------------------------------------------
@pytest.mark.fast
class TestGrokResearchResultParsing:
    """AC1: Test research() returns correctly parsed ResearchResult objects."""

    @pytest.mark.unit
    @pytest.mark.requires_network
    def test_research_returns_research_result_object(self):
        """Research method returns a ResearchResult dataclass."""
        from src.research.grok_client import GrokClient, ResearchResult

        with patch.dict('os.environ', {'XAI_API_KEY': 'test-api-key'}):
            with patch('src.research.grok_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Research content here."}}]
                }
                mock_response.raise_for_status = Mock()
                mock_post.return_value = mock_response

                client = GrokClient(api_key='test-api-key')
                result = client.research("test query")

                assert isinstance(result, ResearchResult)
                assert result.content == "Research content here."
                assert result.topic == "test query"
                assert result.model == "grok-3-latest"

    @pytest.mark.unit
    @pytest.mark.requires_network
    def test_research_result_extracts_sources_from_content(self):
        """Research result extracts source references from content."""
        from src.research.grok_client import GrokClient, ResearchResult

        content_with_sources = """
        According to Reuters, the market is improving.
        Reported by Bloomberg, earnings exceeded expectations.
        Source: official company statement
        https://example.com/report
        @elonmusk said something interesting.
        """

        with patch.dict('os.environ', {'XAI_API_KEY': 'test-api-key'}):
            with patch('src.research.grok_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": content_with_sources}}]
                }
                mock_response.raise_for_status = Mock()
                mock_post.return_value = mock_response

                client = GrokClient(api_key='test-api-key')
                result = client.research("market analysis")

                assert isinstance(result, ResearchResult)
                assert len(result.sources) > 0
                # Should extract Reuters, Bloomberg, URL, and Twitter handle
                sources_lower = [s.lower() for s in result.sources]
                assert any('reuters' in s for s in sources_lower)
                assert any('elonmusk' in s for s in sources_lower)

    @pytest.mark.unit
    @pytest.mark.requires_network
    def test_research_with_quick_depth(self):
        """Research with depth='quick' uses concise prompt."""
        from src.research.grok_client import GrokClient

        with patch.dict('os.environ', {'XAI_API_KEY': 'test-api-key'}):
            with patch('src.research.grok_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Quick summary."}}]
                }
                mock_response.raise_for_status = Mock()
                mock_post.return_value = mock_response

                client = GrokClient(api_key='test-api-key')
                result = client.research("test query", depth="quick")

                # Verify quick depth uses smaller max_tokens
                call_args = mock_post.call_args
                payload = call_args[1]['json']
                assert payload['max_tokens'] == 2000

    @pytest.mark.unit
    @pytest.mark.requires_network
    def test_research_with_comprehensive_depth(self):
        """Research with depth='comprehensive' uses detailed prompt."""
        from src.research.grok_client import GrokClient

        with patch.dict('os.environ', {'XAI_API_KEY': 'test-api-key'}):
            with patch('src.research.grok_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Comprehensive analysis."}}]
                }
                mock_response.raise_for_status = Mock()
                mock_post.return_value = mock_response

                client = GrokClient(api_key='test-api-key')
                result = client.research("test query", depth="comprehensive")

                # Verify comprehensive depth uses larger max_tokens
                call_args = mock_post.call_args
                payload = call_args[1]['json']
                assert payload['max_tokens'] == 6000

    @pytest.mark.unit
    @pytest.mark.requires_network
    def test_research_with_custom_system_prompt(self):
        """Research accepts custom system prompt."""
        from src.research.grok_client import GrokClient

        with patch.dict('os.environ', {'XAI_API_KEY': 'test-api-key'}):
            with patch('src.research.grok_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Custom response."}}]
                }
                mock_response.raise_for_status = Mock()
                mock_post.return_value = mock_response

                client = GrokClient(api_key='test-api-key')
                custom_prompt = "You are a specialized technical writer."
                result = client.research("test query", system_prompt=custom_prompt)

                # Verify custom system prompt was used
                call_args = mock_post.call_args
                payload = call_args[1]['json']
                system_message = payload['messages'][0]
                assert system_message['role'] == 'system'
                assert "specialized technical writer" in system_message['content']


# ---------------------------------------------------------------------------
# AC2: Test GrokClient.research() handles connection errors with retry logic
# ---------------------------------------------------------------------------
@pytest.mark.fast
class TestGrokConnectionErrorHandling:
    """AC2: Test research() handles connection errors with retry logic."""

    @pytest.mark.unit
    @pytest.mark.requires_network
    def test_connection_error_triggers_retry(self):
        """Connection error triggers retry with backoff."""
        from src.research.grok_client import GrokClient

        with patch.dict('os.environ', {'XAI_API_KEY': 'test-api-key'}):
            with patch('src.research.grok_client.time.sleep') as mock_sleep:
                with patch('src.research.grok_client.requests.post') as mock_post:
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "choices": [{"message": {"content": "Success after retry"}}]
                    }
                    mock_response.raise_for_status = Mock()

                    call_count = 0
                    def side_effect(*args, **kwargs):
                        nonlocal call_count
                        call_count += 1
                        if call_count == 1:
                            raise requests.exceptions.ConnectionError("Connection refused")
                        return mock_response

                    mock_post.side_effect = side_effect

                    client = GrokClient(api_key='test-api-key')
                    result = client.research("test query")

                    assert result.content == "Success after retry"
                    assert call_count == 2
                    assert mock_sleep.call_count >= 1

    @pytest.mark.unit
    @pytest.mark.requires_network
    def test_timeout_error_triggers_retry(self):
        """Timeout error triggers retry with backoff."""
        from src.research.grok_client import GrokClient

        with patch.dict('os.environ', {'XAI_API_KEY': 'test-api-key'}):
            with patch('src.research.grok_client.time.sleep') as mock_sleep:
                with patch('src.research.grok_client.requests.post') as mock_post:
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "choices": [{"message": {"content": "Success"}}]
                    }
                    mock_response.raise_for_status = Mock()

                    call_count = 0
                    def side_effect(*args, **kwargs):
                        nonlocal call_count
                        call_count += 1
                        if call_count == 1:
                            raise requests.exceptions.Timeout("Request timed out")
                        return mock_response

                    mock_post.side_effect = side_effect

                    client = GrokClient(api_key='test-api-key')
                    result = client.research("test query")

                    assert result.content == "Success"
                    assert call_count == 2

    @pytest.mark.unit
    @pytest.mark.requires_network
    def test_429_rate_limit_triggers_retry(self):
        """429 Rate limit error triggers retry with backoff."""
        from src.research.grok_client import GrokClient

        with patch.dict('os.environ', {'XAI_API_KEY': 'test-api-key'}):
            with patch('src.research.grok_client.time.sleep') as mock_sleep:
                with patch('src.research.grok_client.requests.post') as mock_post:
                    mock_success_response = Mock()
                    mock_success_response.status_code = 200
                    mock_success_response.json.return_value = {
                        "choices": [{"message": {"content": "Success after rate limit"}}]
                    }
                    mock_success_response.raise_for_status = Mock()

                    mock_rate_limit_response = Mock()
                    mock_rate_limit_response.status_code = 429

                    call_count = 0
                    def side_effect(*args, **kwargs):
                        nonlocal call_count
                        call_count += 1
                        if call_count <= 2:
                            return mock_rate_limit_response
                        return mock_success_response

                    mock_post.side_effect = side_effect

                    client = GrokClient(api_key='test-api-key')
                    result = client.research("test query")

                    assert result.content == "Success after rate limit"
                    assert call_count == 3
                    # Should have slept twice for the two 429 responses
                    assert mock_sleep.call_count == 2

    @pytest.mark.unit
    @pytest.mark.requires_network
    def test_500_server_error_triggers_retry(self):
        """500 Server error triggers retry."""
        from src.research.grok_client import GrokClient

        with patch.dict('os.environ', {'XAI_API_KEY': 'test-api-key'}):
            with patch('src.research.grok_client.time.sleep') as mock_sleep:
                with patch('src.research.grok_client.requests.post') as mock_post:
                    mock_success_response = Mock()
                    mock_success_response.status_code = 200
                    mock_success_response.json.return_value = {
                        "choices": [{"message": {"content": "Success"}}]
                    }
                    mock_success_response.raise_for_status = Mock()

                    mock_error_response = Mock()
                    mock_error_response.status_code = 500

                    call_count = 0
                    def side_effect(*args, **kwargs):
                        nonlocal call_count
                        call_count += 1
                        if call_count == 1:
                            return mock_error_response
                        return mock_success_response

                    mock_post.side_effect = side_effect

                    client = GrokClient(api_key='test-api-key')
                    result = client.research("test query")

                    assert result.content == "Success"
                    assert call_count == 2

    @pytest.mark.unit
    @pytest.mark.requires_network
    def test_max_retries_exceeded_raises_exception(self):
        """After MAX_RETRIES, raises the exception."""
        from src.research.grok_client import GrokClient, MAX_RETRIES

        with patch.dict('os.environ', {'XAI_API_KEY': 'test-api-key'}):
            with patch('src.research.grok_client.time.sleep'):
                with patch('src.research.grok_client.requests.post') as mock_post:
                    mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

                    client = GrokClient(api_key='test-api-key')

                    with pytest.raises(Exception) as exc_info:
                        client.research("test query")

                    assert "failed after" in str(exc_info.value).lower()
                    assert mock_post.call_count == MAX_RETRIES


# ---------------------------------------------------------------------------
# AC3: Test GrokClient.research() handles empty response gracefully
# ---------------------------------------------------------------------------
@pytest.mark.fast
class TestGrokEmptyResponseHandling:
    """AC3: Test research() handles empty response gracefully."""

    @pytest.mark.unit
    @pytest.mark.requires_network
    def test_empty_content_handled(self):
        """Empty content in response returns empty string."""
        from src.research.grok_client import GrokClient

        with patch.dict('os.environ', {'XAI_API_KEY': 'test-api-key'}):
            with patch('src.research.grok_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": ""}}]
                }
                mock_response.raise_for_status = Mock()
                mock_post.return_value = mock_response

                client = GrokClient(api_key='test-api-key')
                result = client.research("test query")

                assert result.content == ""
                assert result.sources == []

    @pytest.mark.unit
    @pytest.mark.requires_network
    def test_whitespace_only_content_handled(self):
        """Whitespace-only content returns that whitespace."""
        from src.research.grok_client import GrokClient

        with patch.dict('os.environ', {'XAI_API_KEY': 'test-api-key'}):
            with patch('src.research.grok_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "   \n  "}}]
                }
                mock_response.raise_for_status = Mock()
                mock_post.return_value = mock_response

                client = GrokClient(api_key='test-api-key')
                result = client.research("test query")

                # Should return the whitespace content as-is
                assert result.content == "   \n  "

    @pytest.mark.unit
    @pytest.mark.requires_network
    def test_no_sources_in_content_returns_empty_list(self):
        """Content without source patterns returns empty sources list."""
        from src.research.grok_client import GrokClient

        with patch.dict('os.environ', {'XAI_API_KEY': 'test-api-key'}):
            with patch('src.research.grok_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "No sources here, just plain text."}}]
                }
                mock_response.raise_for_status = Mock()
                mock_post.return_value = mock_response

                client = GrokClient(api_key='test-api-key')
                result = client.research("test query")

                # No source patterns should result in empty list
                assert result.sources == []

    @pytest.mark.unit
    @pytest.mark.requires_network
    def test_response_without_search_enabled(self):
        """Response works when search is disabled."""
        from src.research.grok_client import GrokClient

        with patch.dict('os.environ', {'XAI_API_KEY': 'test-api-key'}):
            with patch('src.research.grok_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Response without search"}}]
                }
                mock_response.raise_for_status = Mock()
                mock_post.return_value = mock_response

                client = GrokClient(api_key='test-api-key')
                result = client.research("test query", enable_search=False)

                # Verify search was disabled in payload
                call_args = mock_post.call_args
                payload = call_args[1]['json']
                assert 'search' not in payload

    @pytest.mark.unit
    @pytest.mark.requires_network
    def test_sources_deduplicated_and_limited(self):
        """Sources are deduplicated and limited to 15."""
        from src.research.grok_client import GrokClient

        # Content with duplicate sources
        content = """
        According to Reuters, something happened.
        According to Reuters, another thing.
        According to REUTERS, case insensitive.
        """ + " ".join([f"@user{i}" for i in range(20)])

        with patch.dict('os.environ', {'XAI_API_KEY': 'test-api-key'}):
            with patch('src.research.grok_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": content}}]
                }
                mock_response.raise_for_status = Mock()
                mock_post.return_value = mock_response

                client = GrokClient(api_key='test-api-key')
                result = client.research("test query")

                # Should be limited to 15 sources
                assert len(result.sources) <= 15
                # Reuters should only appear once (deduplicated)
                reuters_count = sum(1 for s in result.sources if 'reuters' in s.lower())
                assert reuters_count == 1


# ---------------------------------------------------------------------------
# AC4: Test GrokClient.research() respects configured timeout settings
# ---------------------------------------------------------------------------
@pytest.mark.fast
class TestGrokTimeoutSettings:
    """AC4: Test research() respects configured timeout settings."""

    @pytest.mark.unit
    @pytest.mark.requires_network
    def test_default_timeout_used(self):
        """Default timeout values are used in requests."""
        from src.research.grok_client import GrokClient

        with patch.dict('os.environ', {'XAI_API_KEY': 'test-api-key'}):
            with patch('src.research.grok_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Response"}}]
                }
                mock_response.raise_for_status = Mock()
                mock_post.return_value = mock_response

                client = GrokClient(api_key='test-api-key')
                client.research("test query")

                # Verify timeout was passed
                call_args = mock_post.call_args
                assert 'timeout' in call_args[1]
                # Default is (120, 600) - connect and read timeout
                assert call_args[1]['timeout'] == (120, 600)

    @pytest.mark.unit
    @pytest.mark.requires_network
    def test_retry_backoff_respects_max_backoff(self):
        """Retry backoff is capped at MAX_BACKOFF."""
        from src.research.grok_client import GrokClient, MAX_BACKOFF, MAX_RETRIES

        with patch.dict('os.environ', {'XAI_API_KEY': 'test-api-key'}):
            sleep_times = []
            with patch('src.research.grok_client.time.sleep', side_effect=lambda t: sleep_times.append(t)):
                with patch('src.research.grok_client.random.uniform', return_value=0):
                    with patch('src.research.grok_client.requests.post') as mock_post:
                        mock_post.side_effect = requests.exceptions.Timeout("Timeout")

                        client = GrokClient(api_key='test-api-key')

                        with pytest.raises(Exception):
                            client.research("test query")

                        # All sleep times should be <= MAX_BACKOFF
                        for sleep_time in sleep_times:
                            assert sleep_time <= MAX_BACKOFF

    @pytest.mark.unit
    @pytest.mark.requires_network
    def test_exponential_backoff_pattern(self):
        """Backoff increases exponentially with jitter."""
        from src.research.grok_client import GrokClient, INITIAL_BACKOFF

        with patch.dict('os.environ', {'XAI_API_KEY': 'test-api-key'}):
            sleep_times = []
            with patch('src.research.grok_client.time.sleep', side_effect=lambda t: sleep_times.append(t)):
                with patch('src.research.grok_client.random.uniform', return_value=0.5):
                    with patch('src.research.grok_client.requests.post') as mock_post:
                        mock_success = Mock()
                        mock_success.status_code = 200
                        mock_success.json.return_value = {
                            "choices": [{"message": {"content": "Success"}}]
                        }
                        mock_success.raise_for_status = Mock()

                        call_count = 0
                        def side_effect(*args, **kwargs):
                            nonlocal call_count
                            call_count += 1
                            if call_count <= 3:
                                raise requests.exceptions.Timeout("Timeout")
                            return mock_success

                        mock_post.side_effect = side_effect

                        client = GrokClient(api_key='test-api-key')
                        client.research("test query")

                        # With jitter=0.5:
                        # attempt 0: min(2 * 2^0 + 0.5, 60) = 2.5
                        # attempt 1: min(2 * 2^1 + 0.5, 60) = 4.5
                        # attempt 2: min(2 * 2^2 + 0.5, 60) = 8.5
                        assert len(sleep_times) == 3
                        assert sleep_times[0] == INITIAL_BACKOFF * 1 + 0.5  # 2.5
                        assert sleep_times[1] == INITIAL_BACKOFF * 2 + 0.5  # 4.5
                        assert sleep_times[2] == INITIAL_BACKOFF * 4 + 0.5  # 8.5


# ---------------------------------------------------------------------------
# AC5: Test GrokClient validates input parameters before making requests
# ---------------------------------------------------------------------------
@pytest.mark.fast
class TestGrokInputValidation:
    """AC5: Test GrokClient validates input parameters before making requests."""

    @pytest.mark.unit
    def test_missing_api_key_raises_value_error(self):
        """Missing API key raises ValueError."""
        from src.research.grok_client import GrokClient

        with patch.dict('os.environ', {}, clear=True):
            import os
            if 'XAI_API_KEY' in os.environ:
                del os.environ['XAI_API_KEY']

            with pytest.raises(ValueError) as exc_info:
                GrokClient(api_key=None)

            assert "XAI_API_KEY not set" in str(exc_info.value)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_empty_string_api_key_raises_value_error(self):
        """Empty string API key is treated as missing."""
        from src.research.grok_client import GrokClient

        # Empty string from env should fail - GrokClient checks 'if not self.api_key'
        with patch.dict('os.environ', {'XAI_API_KEY': ''}):
            with pytest.raises(ValueError) as exc_info:
                GrokClient()

            assert "XAI_API_KEY not set" in str(exc_info.value)

    @pytest.mark.unit
    @pytest.mark.requires_network
    def test_api_key_from_parameter_preferred(self):
        """API key from parameter is used over environment variable."""
        from src.research.grok_client import GrokClient

        with patch.dict('os.environ', {'XAI_API_KEY': 'env-api-key'}):
            with patch('src.research.grok_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Response"}}]
                }
                mock_response.raise_for_status = Mock()
                mock_post.return_value = mock_response

                client = GrokClient(api_key='param-api-key')
                client.research("test")

                # Check Authorization header uses param key
                call_args = mock_post.call_args
                headers = call_args[1]['headers']
                assert headers['Authorization'] == 'Bearer param-api-key'

    @pytest.mark.unit
    @pytest.mark.requires_network
    def test_api_key_from_environment_used_when_no_param(self):
        """API key from environment variable is used when no parameter."""
        from src.research.grok_client import GrokClient

        with patch.dict('os.environ', {'XAI_API_KEY': 'env-api-key-12345'}):
            with patch('src.research.grok_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Response"}}]
                }
                mock_response.raise_for_status = Mock()
                mock_post.return_value = mock_response

                client = GrokClient()
                client.research("test")

                # Check Authorization header uses env key
                call_args = mock_post.call_args
                headers = call_args[1]['headers']
                assert headers['Authorization'] == 'Bearer env-api-key-12345'

    @pytest.mark.unit
    @pytest.mark.requires_network
    def test_custom_model_respected(self):
        """Custom model parameter is respected."""
        from src.research.grok_client import GrokClient

        with patch.dict('os.environ', {'XAI_API_KEY': 'test-api-key'}):
            with patch('src.research.grok_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Response"}}]
                }
                mock_response.raise_for_status = Mock()
                mock_post.return_value = mock_response

                client = GrokClient(api_key='test-api-key', model='grok-2-latest')
                result = client.research("test")

                # Verify model in payload
                call_args = mock_post.call_args
                payload = call_args[1]['json']
                assert payload['model'] == 'grok-2-latest'
                assert result.model == 'grok-2-latest'


# ---------------------------------------------------------------------------
# Additional tests for other GrokClient methods
# ---------------------------------------------------------------------------
@pytest.mark.fast
class TestGrokOtherMethods:
    """Additional tests for compare and test_connection methods."""

    @pytest.mark.unit
    @pytest.mark.requires_network
    def test_compare_returns_research_result(self):
        """compare method returns ResearchResult."""
        from src.research.grok_client import GrokClient, ResearchResult

        with patch.dict('os.environ', {'XAI_API_KEY': 'test-api-key'}):
            with patch('src.research.grok_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Comparison results"}}]
                }
                mock_response.raise_for_status = Mock()
                mock_post.return_value = mock_response

                client = GrokClient(api_key='test-api-key')
                result = client.compare(["Item A", "Item B"])

                assert isinstance(result, ResearchResult)
                assert "Item A" in result.topic
                assert "Item B" in result.topic

    @pytest.mark.unit
    @pytest.mark.requires_network
    def test_compare_with_criteria(self):
        """compare method accepts optional criteria."""
        from src.research.grok_client import GrokClient

        with patch.dict('os.environ', {'XAI_API_KEY': 'test-api-key'}):
            with patch('src.research.grok_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Comparison with criteria"}}]
                }
                mock_response.raise_for_status = Mock()
                mock_post.return_value = mock_response

                client = GrokClient(api_key='test-api-key')
                result = client.compare(
                    ["Python", "JavaScript"],
                    criteria=["performance", "ease of learning"]
                )

                # Verify criteria were included in the prompt
                call_args = mock_post.call_args
                payload = call_args[1]['json']
                user_content = payload['messages'][0]['content']
                assert "performance" in user_content
                assert "ease of learning" in user_content

    @pytest.mark.unit
    @pytest.mark.requires_network
    def test_test_connection_success(self):
        """test_connection returns True on successful API call."""
        from src.research.grok_client import GrokClient

        with patch.dict('os.environ', {'XAI_API_KEY': 'test-api-key'}):
            with patch('src.research.grok_client.requests.get') as mock_get:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.raise_for_status = Mock()
                mock_get.return_value = mock_response

                client = GrokClient(api_key='test-api-key')
                result = client.test_connection()

                assert result is True
                mock_get.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.requires_network
    def test_test_connection_failure(self):
        """test_connection returns False on API error."""
        from src.research.grok_client import GrokClient

        with patch.dict('os.environ', {'XAI_API_KEY': 'test-api-key'}):
            with patch('src.research.grok_client.requests.get') as mock_get:
                mock_get.side_effect = requests.exceptions.ConnectionError("Failed")

                client = GrokClient(api_key='test-api-key')
                result = client.test_connection()

                assert result is False

    @pytest.mark.unit
    @pytest.mark.requires_network
    def test_compare_enables_search(self):
        """compare method enables search by default."""
        from src.research.grok_client import GrokClient

        with patch.dict('os.environ', {'XAI_API_KEY': 'test-api-key'}):
            with patch('src.research.grok_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Comparison"}}]
                }
                mock_response.raise_for_status = Mock()
                mock_post.return_value = mock_response

                client = GrokClient(api_key='test-api-key')
                client.compare(["A", "B"])

                # Verify search was enabled
                call_args = mock_post.call_args
                payload = call_args[1]['json']
                assert payload.get('search', {}).get('enabled') is True


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------
@pytest.mark.fast
class TestGrokEdgeCases:
    """Edge case tests for GrokClient."""

    @pytest.mark.unit
    @pytest.mark.requires_network
    def test_http_error_with_json_body(self):
        """HTTP error with JSON body includes error details."""
        from src.research.grok_client import GrokClient

        with patch.dict('os.environ', {'XAI_API_KEY': 'test-api-key'}):
            with patch('src.research.grok_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 400
                mock_response.json.return_value = {"error": "Invalid request"}
                mock_response.text = '{"error": "Invalid request"}'
                mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("400 Bad Request")
                mock_post.return_value = mock_response

                client = GrokClient(api_key='test-api-key')

                with pytest.raises(Exception) as exc_info:
                    client.research("test query")

                assert "Invalid request" in str(exc_info.value) or "400" in str(exc_info.value)

    @pytest.mark.unit
    @pytest.mark.requires_network
    def test_http_error_with_text_body(self):
        """HTTP error with non-JSON body handles gracefully."""
        from src.research.grok_client import GrokClient

        with patch.dict('os.environ', {'XAI_API_KEY': 'test-api-key'}):
            with patch('src.research.grok_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 400
                mock_response.json.side_effect = ValueError("Not JSON")
                mock_response.text = "Plain text error"
                mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("400 Bad Request")
                mock_post.return_value = mock_response

                client = GrokClient(api_key='test-api-key')

                with pytest.raises(Exception) as exc_info:
                    client.research("test query")

                # Should include the text error or status code
                assert "Plain text error" in str(exc_info.value) or "400" in str(exc_info.value)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_source_extraction_patterns(self):
        """Source extraction handles various patterns correctly."""
        from src.research.grok_client import GrokClient

        content = """
        According to The Wall Street Journal, markets rose.
        reported by BBC News, the event occurred.
        Source: company press release
        [Citation needed]
        @TwitterUser shared this.
        Check https://example.com/article and https://other.com/page
        """

        with patch.dict('os.environ', {'XAI_API_KEY': 'test-api-key'}):
            client = GrokClient(api_key='test-api-key')
            sources = client._extract_sources(content)

            # Should have found multiple sources
            assert len(sources) > 0
            assert len(sources) <= 15  # Capped at 15

            # Should find various patterns
            sources_str = " ".join(sources).lower()
            assert 'wall street journal' in sources_str or 'bbc' in sources_str

    @pytest.mark.unit
    @pytest.mark.requires_network
    def test_research_search_payload_structure(self):
        """Research method creates correct payload structure."""
        from src.research.grok_client import GrokClient

        with patch.dict('os.environ', {'XAI_API_KEY': 'test-api-key'}):
            with patch('src.research.grok_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Response"}}]
                }
                mock_response.raise_for_status = Mock()
                mock_post.return_value = mock_response

                client = GrokClient(api_key='test-api-key')
                client.research("test query", enable_search=True)

                call_args = mock_post.call_args
                payload = call_args[1]['json']

                # Verify payload structure
                assert 'model' in payload
                assert 'messages' in payload
                assert 'temperature' in payload
                assert 'max_tokens' in payload
                assert payload['search'] == {'enabled': True}

                # Verify messages structure
                assert len(payload['messages']) == 2
                assert payload['messages'][0]['role'] == 'system'
                assert payload['messages'][1]['role'] == 'user'

    @pytest.mark.unit
    @pytest.mark.requires_network
    def test_base_url_used_correctly(self):
        """API calls use correct base URL."""
        from src.research.grok_client import GrokClient

        with patch.dict('os.environ', {'XAI_API_KEY': 'test-api-key'}):
            with patch('src.research.grok_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Response"}}]
                }
                mock_response.raise_for_status = Mock()
                mock_post.return_value = mock_response

                client = GrokClient(api_key='test-api-key')
                client.research("test query")

                call_args = mock_post.call_args
                url = call_args[0][0]
                assert url == "https://api.x.ai/v1/chat/completions"
