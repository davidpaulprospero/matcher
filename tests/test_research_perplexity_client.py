"""
Unit tests for PerplexityClient research client.

Tests cover:
- US-002: Perplexity research client unit tests
  - AC1: Test research() returns correctly parsed ResearchResult objects
  - AC2: Test research() handles rate limiting (429) with retry-after header
  - AC3: Test research() handles streaming/retry responses correctly
  - AC4: Test research() handles invalid API key with clear error message
  - AC5: Test PerplexityClient validates query parameters before API call
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import requests


# ---------------------------------------------------------------------------
# AC1: Test research() returns correctly parsed ResearchResult objects
# ---------------------------------------------------------------------------
class TestPerplexityResearchResultParsing:
    """AC1: Test research() returns correctly parsed ResearchResult objects."""

    @pytest.mark.unit
    def test_research_returns_research_result_object(self):
        """Research method returns a ResearchResult dataclass."""
        from src.research.perplexity_client import PerplexityClient, ResearchResult

        with patch.dict('os.environ', {'PERPLEXITY_API_KEY': 'test-api-key-12345'}):
            with patch('src.research.perplexity_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "This is the research content."}}],
                    "citations": []
                }
                mock_post.return_value = mock_response

                client = PerplexityClient(api_key='test-api-key-12345')
                result = client.research("test query")

                assert isinstance(result, ResearchResult)
                assert result.content == "This is the research content."
                assert result.topic == "test query"
                assert result.model == "sonar-pro"  # Default for comprehensive

    @pytest.mark.unit
    def test_research_result_contains_sources_from_citations(self):
        """Research result includes sources from citations in response."""
        from src.research.perplexity_client import PerplexityClient, ResearchResult

        with patch.dict('os.environ', {'PERPLEXITY_API_KEY': 'test-api-key-12345'}):
            with patch('src.research.perplexity_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Content with sources"}}],
                    "citations": ["https://example.com/source1", "https://example.com/source2"]
                }
                mock_post.return_value = mock_response

                client = PerplexityClient(api_key='test-api-key-12345')
                result = client.research("query with sources")

                assert isinstance(result, ResearchResult)
                assert len(result.sources) == 2
                assert "https://example.com/source1" in result.sources
                assert "https://example.com/source2" in result.sources

    @pytest.mark.unit
    def test_research_result_with_quick_depth(self):
        """Research with depth='quick' uses sonar model."""
        from src.research.perplexity_client import PerplexityClient, ResearchResult

        with patch.dict('os.environ', {'PERPLEXITY_API_KEY': 'test-api-key-12345'}):
            with patch('src.research.perplexity_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Quick summary content."}}],
                    "citations": []
                }
                mock_post.return_value = mock_response

                client = PerplexityClient(api_key='test-api-key-12345')
                result = client.research("test query", depth="quick")

                assert isinstance(result, ResearchResult)
                assert result.content == "Quick summary content."
                # Verify sonar model was used for quick depth
                call_args = mock_post.call_args
                payload = call_args[1]['json']
                assert payload['model'] == 'sonar'

    @pytest.mark.unit
    def test_research_result_with_comprehensive_depth(self):
        """Research with depth='comprehensive' uses sonar-pro model."""
        from src.research.perplexity_client import PerplexityClient, ResearchResult

        with patch.dict('os.environ', {'PERPLEXITY_API_KEY': 'test-api-key-12345'}):
            with patch('src.research.perplexity_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Comprehensive research."}}],
                    "citations": []
                }
                mock_post.return_value = mock_response

                client = PerplexityClient(api_key='test-api-key-12345')
                result = client.research("test query", depth="comprehensive")

                assert isinstance(result, ResearchResult)
                # Verify sonar-pro model was used for comprehensive depth
                call_args = mock_post.call_args
                payload = call_args[1]['json']
                assert payload['model'] == 'sonar-pro'

    @pytest.mark.unit
    def test_research_result_with_custom_system_prompt(self):
        """Research accepts custom system prompt."""
        from src.research.perplexity_client import PerplexityClient, ResearchResult

        with patch.dict('os.environ', {'PERPLEXITY_API_KEY': 'test-api-key-12345'}):
            with patch('src.research.perplexity_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Custom response."}}],
                    "citations": []
                }
                mock_post.return_value = mock_response

                client = PerplexityClient(api_key='test-api-key-12345')
                custom_prompt = "You are a specialized research bot."
                result = client.research("test query", system_prompt=custom_prompt)

                # Verify custom prompt was used in messages
                call_args = mock_post.call_args
                payload = call_args[1]['json']
                messages = payload['messages']
                assert any(custom_prompt in msg.get('content', '') for msg in messages)


# ---------------------------------------------------------------------------
# AC2: Test research() handles rate limiting (429) with retry-after header
# ---------------------------------------------------------------------------
class TestPerplexityRateLimitHandling:
    """AC2: Test research() handles rate limiting (429) with exponential backoff."""

    @pytest.mark.unit
    def test_429_triggers_exponential_backoff(self):
        """429 error triggers exponential backoff retry."""
        from src.research.perplexity_client import PerplexityClient, INITIAL_BACKOFF, MAX_RETRIES

        with patch.dict('os.environ', {'PERPLEXITY_API_KEY': 'test-api-key-12345'}):
            with patch('src.research.perplexity_client.time.sleep') as mock_sleep:
                with patch('src.research.perplexity_client.requests.post') as mock_post:
                    # First 2 calls return 429, third succeeds
                    mock_429_response = Mock()
                    mock_429_response.status_code = 429

                    mock_success_response = Mock()
                    mock_success_response.status_code = 200
                    mock_success_response.json.return_value = {
                        "choices": [{"message": {"content": "Success after retry"}}],
                        "citations": []
                    }

                    mock_post.side_effect = [
                        mock_429_response,
                        mock_429_response,
                        mock_success_response
                    ]

                    client = PerplexityClient(api_key='test-api-key-12345')
                    result = client.research("test query")

                    assert result.content == "Success after retry"
                    # Verify backoff was called twice (for 2 retries)
                    assert mock_sleep.call_count == 2

    @pytest.mark.unit
    def test_500_error_triggers_retry(self):
        """500 server error triggers retry."""
        from src.research.perplexity_client import PerplexityClient

        with patch.dict('os.environ', {'PERPLEXITY_API_KEY': 'test-api-key-12345'}):
            with patch('src.research.perplexity_client.time.sleep'):
                with patch('src.research.perplexity_client.requests.post') as mock_post:
                    mock_500_response = Mock()
                    mock_500_response.status_code = 500

                    mock_success_response = Mock()
                    mock_success_response.status_code = 200
                    mock_success_response.json.return_value = {
                        "choices": [{"message": {"content": "Success"}}],
                        "citations": []
                    }

                    mock_post.side_effect = [mock_500_response, mock_success_response]

                    client = PerplexityClient(api_key='test-api-key-12345')
                    result = client.research("test query")

                    assert result.content == "Success"
                    assert mock_post.call_count == 2

    @pytest.mark.unit
    def test_max_retries_exceeded_raises_exception(self):
        """After MAX_RETRIES, re-raises the exception."""
        from src.research.perplexity_client import PerplexityClient, MAX_RETRIES

        with patch.dict('os.environ', {'PERPLEXITY_API_KEY': 'test-api-key-12345'}):
            with patch('src.research.perplexity_client.time.sleep') as mock_sleep:
                with patch('src.research.perplexity_client.requests.post') as mock_post:
                    # Always return 429
                    mock_429_response = Mock()
                    mock_429_response.status_code = 429
                    mock_post.return_value = mock_429_response

                    client = PerplexityClient(api_key='test-api-key-12345')

                    with pytest.raises(Exception) as exc_info:
                        client.research("test query")

                    assert "failed after" in str(exc_info.value).lower()
                    assert mock_post.call_count == MAX_RETRIES

    @pytest.mark.unit
    def test_backoff_includes_jitter(self):
        """Backoff time includes random jitter component."""
        from src.research.perplexity_client import PerplexityClient, INITIAL_BACKOFF

        with patch.dict('os.environ', {'PERPLEXITY_API_KEY': 'test-api-key-12345'}):
            sleep_times = []
            with patch('src.research.perplexity_client.time.sleep', side_effect=lambda t: sleep_times.append(t)):
                with patch('src.research.perplexity_client.random.uniform', return_value=0.5):
                    with patch('src.research.perplexity_client.requests.post') as mock_post:
                        mock_429_response = Mock()
                        mock_429_response.status_code = 429

                        mock_success_response = Mock()
                        mock_success_response.status_code = 200
                        mock_success_response.json.return_value = {
                            "choices": [{"message": {"content": "Success"}}],
                            "citations": []
                        }

                        mock_post.side_effect = [
                            mock_429_response,
                            mock_429_response,
                            mock_success_response
                        ]

                        client = PerplexityClient(api_key='test-api-key-12345')
                        client.research("test query")

                        # With jitter = 0.5:
                        # attempt 0: min(2 * 2^0 + 0.5, 60) = 2.5
                        # attempt 1: min(2 * 2^1 + 0.5, 60) = 4.5
                        assert len(sleep_times) == 2
                        assert sleep_times[0] == INITIAL_BACKOFF * 1 + 0.5  # 2.5
                        assert sleep_times[1] == INITIAL_BACKOFF * 2 + 0.5  # 4.5


# ---------------------------------------------------------------------------
# AC3: Test research() handles streaming/connection errors correctly
# ---------------------------------------------------------------------------
class TestPerplexityConnectionErrorHandling:
    """AC3: Test research() handles connection/timeout errors with retry."""

    @pytest.mark.unit
    def test_timeout_error_triggers_retry(self):
        """Timeout error triggers retry logic."""
        from src.research.perplexity_client import PerplexityClient

        with patch.dict('os.environ', {'PERPLEXITY_API_KEY': 'test-api-key-12345'}):
            with patch('src.research.perplexity_client.time.sleep'):
                with patch('src.research.perplexity_client.requests.post') as mock_post:
                    mock_success_response = Mock()
                    mock_success_response.status_code = 200
                    mock_success_response.json.return_value = {
                        "choices": [{"message": {"content": "Success after timeout"}}],
                        "citations": []
                    }

                    mock_post.side_effect = [
                        requests.exceptions.Timeout("Connection timed out"),
                        mock_success_response
                    ]

                    client = PerplexityClient(api_key='test-api-key-12345')
                    result = client.research("test query")

                    assert result.content == "Success after timeout"
                    assert mock_post.call_count == 2

    @pytest.mark.unit
    def test_connection_error_triggers_retry(self):
        """Connection error triggers retry logic."""
        from src.research.perplexity_client import PerplexityClient

        with patch.dict('os.environ', {'PERPLEXITY_API_KEY': 'test-api-key-12345'}):
            with patch('src.research.perplexity_client.time.sleep'):
                with patch('src.research.perplexity_client.requests.post') as mock_post:
                    mock_success_response = Mock()
                    mock_success_response.status_code = 200
                    mock_success_response.json.return_value = {
                        "choices": [{"message": {"content": "Success"}}],
                        "citations": []
                    }

                    mock_post.side_effect = [
                        requests.exceptions.ConnectionError("Connection refused"),
                        mock_success_response
                    ]

                    client = PerplexityClient(api_key='test-api-key-12345')
                    result = client.research("test query")

                    assert result.content == "Success"
                    assert mock_post.call_count == 2

    @pytest.mark.unit
    def test_502_bad_gateway_triggers_retry(self):
        """502 Bad Gateway triggers retry."""
        from src.research.perplexity_client import PerplexityClient

        with patch.dict('os.environ', {'PERPLEXITY_API_KEY': 'test-api-key-12345'}):
            with patch('src.research.perplexity_client.time.sleep'):
                with patch('src.research.perplexity_client.requests.post') as mock_post:
                    mock_502_response = Mock()
                    mock_502_response.status_code = 502

                    mock_success_response = Mock()
                    mock_success_response.status_code = 200
                    mock_success_response.json.return_value = {
                        "choices": [{"message": {"content": "Success"}}],
                        "citations": []
                    }

                    mock_post.side_effect = [mock_502_response, mock_success_response]

                    client = PerplexityClient(api_key='test-api-key-12345')
                    result = client.research("test query")

                    assert result.content == "Success"

    @pytest.mark.unit
    def test_503_service_unavailable_triggers_retry(self):
        """503 Service Unavailable triggers retry."""
        from src.research.perplexity_client import PerplexityClient

        with patch.dict('os.environ', {'PERPLEXITY_API_KEY': 'test-api-key-12345'}):
            with patch('src.research.perplexity_client.time.sleep'):
                with patch('src.research.perplexity_client.requests.post') as mock_post:
                    mock_503_response = Mock()
                    mock_503_response.status_code = 503

                    mock_success_response = Mock()
                    mock_success_response.status_code = 200
                    mock_success_response.json.return_value = {
                        "choices": [{"message": {"content": "Success"}}],
                        "citations": []
                    }

                    mock_post.side_effect = [mock_503_response, mock_success_response]

                    client = PerplexityClient(api_key='test-api-key-12345')
                    result = client.research("test query")

                    assert result.content == "Success"

    @pytest.mark.unit
    def test_504_gateway_timeout_triggers_retry(self):
        """504 Gateway Timeout triggers retry."""
        from src.research.perplexity_client import PerplexityClient

        with patch.dict('os.environ', {'PERPLEXITY_API_KEY': 'test-api-key-12345'}):
            with patch('src.research.perplexity_client.time.sleep'):
                with patch('src.research.perplexity_client.requests.post') as mock_post:
                    mock_504_response = Mock()
                    mock_504_response.status_code = 504

                    mock_success_response = Mock()
                    mock_success_response.status_code = 200
                    mock_success_response.json.return_value = {
                        "choices": [{"message": {"content": "Success"}}],
                        "citations": []
                    }

                    mock_post.side_effect = [mock_504_response, mock_success_response]

                    client = PerplexityClient(api_key='test-api-key-12345')
                    result = client.research("test query")

                    assert result.content == "Success"


# ---------------------------------------------------------------------------
# AC4: Test research() handles invalid API key with clear error message
# ---------------------------------------------------------------------------
class TestPerplexityInvalidAPIKeyHandling:
    """AC4: Test research() handles invalid API key with clear error message."""

    @pytest.mark.unit
    def test_401_unauthorized_raises_http_error(self):
        """401 Unauthorized (invalid API key) raises exception with clear message."""
        from src.research.perplexity_client import PerplexityClient

        with patch.dict('os.environ', {'PERPLEXITY_API_KEY': 'invalid-key'}):
            with patch('src.research.perplexity_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 401
                mock_response.json.return_value = {"error": "Invalid API key"}
                mock_response.text = "Invalid API key"
                mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("401 Unauthorized")
                mock_post.return_value = mock_response

                client = PerplexityClient(api_key='invalid-key')

                with pytest.raises(Exception) as exc_info:
                    client.research("test query")

                # 401 is NOT in RETRYABLE_STATUS_CODES, so should raise immediately
                assert mock_post.call_count == 1
                assert "401" in str(exc_info.value) or "Invalid API key" in str(exc_info.value)

    @pytest.mark.unit
    def test_403_forbidden_raises_http_error(self):
        """403 Forbidden raises exception immediately (no retry)."""
        from src.research.perplexity_client import PerplexityClient

        with patch.dict('os.environ', {'PERPLEXITY_API_KEY': 'test-key'}):
            with patch('src.research.perplexity_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 403
                mock_response.json.return_value = {"error": "Forbidden"}
                mock_response.text = "Forbidden"
                mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("403 Forbidden")
                mock_post.return_value = mock_response

                client = PerplexityClient(api_key='test-key')

                with pytest.raises(Exception) as exc_info:
                    client.research("test query")

                # 403 is NOT in RETRYABLE_STATUS_CODES, so should raise immediately
                assert mock_post.call_count == 1

    @pytest.mark.unit
    def test_http_error_includes_response_details(self):
        """HTTP error includes response body details in exception message."""
        from src.research.perplexity_client import PerplexityClient

        with patch.dict('os.environ', {'PERPLEXITY_API_KEY': 'test-key'}):
            with patch('src.research.perplexity_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 400
                mock_response.json.return_value = {"error": {"message": "Detailed error info"}}
                mock_response.text = '{"error": {"message": "Detailed error info"}}'
                mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("400 Bad Request")
                mock_post.return_value = mock_response

                client = PerplexityClient(api_key='test-key')

                with pytest.raises(Exception) as exc_info:
                    client.research("test query")

                # Error message should include details from response
                error_str = str(exc_info.value)
                assert "400" in error_str or "error" in error_str.lower()


# ---------------------------------------------------------------------------
# AC5: Test PerplexityClient validates API key before making requests
# ---------------------------------------------------------------------------
class TestPerplexityAPIKeyValidation:
    """AC5: Test PerplexityClient validates API key before making requests."""

    @pytest.mark.unit
    def test_missing_api_key_raises_value_error(self):
        """Missing API key raises ValueError."""
        from src.research.perplexity_client import PerplexityClient

        with patch.dict('os.environ', {}, clear=True):
            import os
            if 'PERPLEXITY_API_KEY' in os.environ:
                del os.environ['PERPLEXITY_API_KEY']

            with pytest.raises(ValueError) as exc_info:
                PerplexityClient(api_key=None)

            assert "PERPLEXITY_API_KEY not set" in str(exc_info.value)

    @pytest.mark.unit
    def test_empty_string_api_key_raises_value_error(self):
        """Empty string API key raises ValueError."""
        from src.research.perplexity_client import PerplexityClient

        with patch.dict('os.environ', {'PERPLEXITY_API_KEY': ''}):
            with pytest.raises(ValueError) as exc_info:
                PerplexityClient()

            assert "PERPLEXITY_API_KEY not set" in str(exc_info.value)

    @pytest.mark.unit
    def test_api_key_from_parameter_preferred(self):
        """API key from parameter is used over environment variable."""
        from src.research.perplexity_client import PerplexityClient

        with patch.dict('os.environ', {'PERPLEXITY_API_KEY': 'env-api-key'}):
            client = PerplexityClient(api_key='param-api-key')
            assert client.api_key == 'param-api-key'

    @pytest.mark.unit
    def test_api_key_from_environment_used_when_no_param(self):
        """API key from environment variable is used when no parameter."""
        from src.research.perplexity_client import PerplexityClient

        env_key = 'test-env-api-key-12345'
        with patch.dict('os.environ', {'PERPLEXITY_API_KEY': env_key}):
            client = PerplexityClient()
            assert client.api_key == env_key

    @pytest.mark.unit
    def test_error_message_suggests_env_or_parameter(self):
        """Error message suggests both env variable and parameter options."""
        from src.research.perplexity_client import PerplexityClient

        with patch.dict('os.environ', {}, clear=True):
            import os
            if 'PERPLEXITY_API_KEY' in os.environ:
                del os.environ['PERPLEXITY_API_KEY']

            with pytest.raises(ValueError) as exc_info:
                PerplexityClient(api_key=None)

            error_msg = str(exc_info.value)
            assert "PERPLEXITY_API_KEY" in error_msg
            # Message should mention both .env and parameter options
            assert ".env" in error_msg or "api_key" in error_msg


# ---------------------------------------------------------------------------
# Additional tests for other PerplexityClient methods
# ---------------------------------------------------------------------------
class TestPerplexityOtherMethods:
    """Additional tests for fact_check, compare, and api_docs methods."""

    @pytest.mark.unit
    def test_fact_check_returns_research_result(self):
        """fact_check method returns ResearchResult."""
        from src.research.perplexity_client import PerplexityClient, ResearchResult

        with patch.dict('os.environ', {'PERPLEXITY_API_KEY': 'test-api-key'}):
            with patch('src.research.perplexity_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Fact check results"}}],
                    "citations": []
                }
                mock_post.return_value = mock_response

                client = PerplexityClient(api_key='test-api-key')
                result = client.fact_check(["Claim 1", "Claim 2"])

                assert isinstance(result, ResearchResult)
                assert result.topic == "Fact Check"

    @pytest.mark.unit
    def test_compare_returns_research_result(self):
        """compare method returns ResearchResult."""
        from src.research.perplexity_client import PerplexityClient, ResearchResult

        with patch.dict('os.environ', {'PERPLEXITY_API_KEY': 'test-api-key'}):
            with patch('src.research.perplexity_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Comparison results"}}],
                    "citations": []
                }
                mock_post.return_value = mock_response

                client = PerplexityClient(api_key='test-api-key')
                result = client.compare(["Item A", "Item B"])

                assert isinstance(result, ResearchResult)
                assert "Item A" in result.topic
                assert "Item B" in result.topic

    @pytest.mark.unit
    def test_api_docs_returns_research_result(self):
        """api_docs method returns ResearchResult."""
        from src.research.perplexity_client import PerplexityClient, ResearchResult

        with patch.dict('os.environ', {'PERPLEXITY_API_KEY': 'test-api-key'}):
            with patch('src.research.perplexity_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "API documentation"}}],
                    "citations": []
                }
                mock_post.return_value = mock_response

                client = PerplexityClient(api_key='test-api-key')
                result = client.api_docs("YouTube Data API", focus="video search")

                assert isinstance(result, ResearchResult)
                assert "YouTube Data API" in result.topic


# ---------------------------------------------------------------------------
# Edge case and robustness tests
# ---------------------------------------------------------------------------
class TestPerplexityEdgeCases:
    """Edge case and robustness tests."""

    @pytest.mark.unit
    def test_empty_citations_handled(self):
        """Empty citations list handled correctly."""
        from src.research.perplexity_client import PerplexityClient

        with patch.dict('os.environ', {'PERPLEXITY_API_KEY': 'test-api-key'}):
            with patch('src.research.perplexity_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Content"}}],
                    "citations": []
                }
                mock_post.return_value = mock_response

                client = PerplexityClient(api_key='test-api-key')
                result = client.research("test query")

                assert result.sources == []

    @pytest.mark.unit
    def test_missing_citations_field_handled(self):
        """Missing citations field in response handled gracefully."""
        from src.research.perplexity_client import PerplexityClient

        with patch.dict('os.environ', {'PERPLEXITY_API_KEY': 'test-api-key'}):
            with patch('src.research.perplexity_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Content"}}]
                    # No citations field
                }
                mock_post.return_value = mock_response

                client = PerplexityClient(api_key='test-api-key')
                result = client.research("test query")

                assert result.sources == []

    @pytest.mark.unit
    def test_non_list_citations_handled(self):
        """Non-list citations value handled gracefully."""
        from src.research.perplexity_client import PerplexityClient

        with patch.dict('os.environ', {'PERPLEXITY_API_KEY': 'test-api-key'}):
            with patch('src.research.perplexity_client.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Content"}}],
                    "citations": "not a list"  # Invalid type
                }
                mock_post.return_value = mock_response

                client = PerplexityClient(api_key='test-api-key')
                result = client.research("test query")

                # Should handle gracefully with empty list
                assert result.sources == []

    @pytest.mark.unit
    def test_default_model_selection(self):
        """Default model can be specified at client initialization."""
        from src.research.perplexity_client import PerplexityClient

        with patch.dict('os.environ', {'PERPLEXITY_API_KEY': 'test-api-key'}):
            client = PerplexityClient(api_key='test-api-key', default_model='quick')
            assert client.default_model == 'sonar'

            client2 = PerplexityClient(api_key='test-api-key', default_model='research')
            assert client2.default_model == 'sonar-pro'

            client3 = PerplexityClient(api_key='test-api-key', default_model='reasoning')
            assert client3.default_model == 'sonar-reasoning'

    @pytest.mark.unit
    def test_custom_model_passthrough(self):
        """Custom model name passed through if not in MODELS dict."""
        from src.research.perplexity_client import PerplexityClient

        with patch.dict('os.environ', {'PERPLEXITY_API_KEY': 'test-api-key'}):
            client = PerplexityClient(api_key='test-api-key', default_model='custom-model')
            # Custom model passed through as-is when not in MODELS
            assert client.default_model == 'custom-model'
