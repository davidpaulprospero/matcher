"""
Tests for WatcherAgent error classification.

US-004: Add agents watcher unit tests

Tests for src/agents/watcher.py covering:
- ErrorClassification dataclass
- WatcherAgent.classify_error() behavior
- Error category handling (API, disk, network)
- Escalation logic
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.watcher import WatcherAgent, ErrorClassification, WATCHER_PROMPT


class TestErrorClassificationDataclass:
    """Tests for ErrorClassification dataclass."""

    def test_error_classification_all_fields(self):
        """Test ErrorClassification accepts all required fields."""
        classification = ErrorClassification(
            category="api",
            severity="recoverable",
            suggested_healer="api-healer",
            confidence=0.9,
            needs_llm_healer=False,
            reasoning="Rate limit detected"
        )

        assert classification.category == "api"
        assert classification.severity == "recoverable"
        assert classification.suggested_healer == "api-healer"
        assert classification.confidence == 0.9
        assert classification.needs_llm_healer is False
        assert classification.reasoning == "Rate limit detected"

    def test_error_classification_categories(self):
        """Test ErrorClassification accepts valid categories."""
        valid_categories = ["api", "disk", "path", "checkpoint", "download", "otio", "config", "unknown"]

        for category in valid_categories:
            classification = ErrorClassification(
                category=category,
                severity="recoverable",
                suggested_healer="",
                confidence=0.5,
                needs_llm_healer=False,
                reasoning=""
            )
            assert classification.category == category

    def test_error_classification_severities(self):
        """Test ErrorClassification accepts valid severities."""
        valid_severities = ["critical", "recoverable", "transient"]

        for severity in valid_severities:
            classification = ErrorClassification(
                category="unknown",
                severity=severity,
                suggested_healer="",
                confidence=0.5,
                needs_llm_healer=False,
                reasoning=""
            )
            assert classification.severity == severity


class TestWatcherAgentInit:
    """Tests for WatcherAgent initialization."""

    def test_watcher_agent_init_defaults(self):
        """Test WatcherAgent uses default values from config."""
        mock_config = Mock()
        mock_config.provider = "ollama"
        mock_config.model = "llama3.2"
        mock_config.fallback_model = "llama3.1"
        mock_config.host = "http://localhost:11434"
        mock_config.timeout = 30.0
        mock_config.escalate_threshold = 0.7

        agent = WatcherAgent(config=mock_config)

        assert agent.provider == "ollama"
        assert agent.model == "llama3.2"
        assert agent.fallback_model == "llama3.1"
        assert agent.host == "http://localhost:11434"
        assert agent.timeout == 30.0
        assert agent.escalate_threshold == 0.7
        assert agent.client is None
        assert agent._initialized is False

    def test_watcher_agent_init_missing_config_attrs(self):
        """Test WatcherAgent handles missing config attributes with defaults."""
        mock_config = Mock(spec=[])  # Empty spec - no attributes

        agent = WatcherAgent(config=mock_config)

        # Should use defaults from getattr
        assert agent.provider == "ollama"
        assert agent.model == "llama3.2"
        assert agent.timeout == 30.0

    def test_watcher_agent_init_with_project_dir(self, tmp_path):
        """Test WatcherAgent accepts project_dir parameter."""
        mock_config = Mock()
        project_dir = tmp_path / "test_project"

        agent = WatcherAgent(config=mock_config, project_dir=project_dir)

        assert agent.project_dir == project_dir

    def test_watcher_agent_init_with_healing_logger(self):
        """Test WatcherAgent accepts healing_logger parameter."""
        mock_config = Mock()
        mock_logger = Mock()

        agent = WatcherAgent(config=mock_config, healing_logger=mock_logger)

        assert agent.healing_logger == mock_logger

    def test_watcher_agent_init_with_fallback_chain(self):
        """Test WatcherAgent accepts fallback_chain parameter."""
        mock_config = Mock()
        mock_chain = Mock()

        agent = WatcherAgent(config=mock_config, fallback_chain=mock_chain)

        assert agent.fallback_chain == mock_chain


class TestWatcherAgentClassifyError:
    """Tests for WatcherAgent.classify_error()."""

    @pytest.fixture
    def mock_watcher_config(self):
        """Create mock watcher config."""
        config = Mock()
        config.provider = "ollama"
        config.model = "llama3.2"
        config.fallback_model = "llama3.1"
        config.host = "http://localhost:11434"
        config.timeout = 30.0
        config.escalate_threshold = 0.7
        return config

    def test_classify_error_returns_error_classification(self, mock_watcher_config):
        """Test classify_error() returns ErrorClassification on success."""
        agent = WatcherAgent(config=mock_watcher_config)

        # Mock the client
        mock_response = Mock()
        mock_response.parsed_data = {
            "category": "api",
            "severity": "transient",
            "suggested_healer": "api-healer",
            "confidence": 0.85,
            "needs_llm_healer": False,
            "reasoning": "Rate limit detected"
        }

        mock_client = Mock()
        mock_client.generate.return_value = mock_response
        agent.client = mock_client
        agent._initialized = True

        error = Exception("429 Too Many Requests")
        context = {"stage": "DOWNLOAD"}

        result = agent.classify_error(error, context)

        assert result is not None
        assert isinstance(result, ErrorClassification)
        assert result.category == "api"
        assert result.severity == "transient"
        assert result.confidence == 0.85

    def test_classify_error_api_rate_limit(self, mock_watcher_config):
        """Test classify_error() handles API rate limit errors."""
        agent = WatcherAgent(config=mock_watcher_config)

        mock_response = Mock()
        mock_response.parsed_data = {
            "category": "api",
            "severity": "transient",
            "suggested_healer": "api-healer",
            "confidence": 0.92,
            "needs_llm_healer": False,
            "reasoning": "HTTP 429 rate limit - transient, retry after backoff"
        }

        mock_client = Mock()
        mock_client.generate.return_value = mock_response
        agent.client = mock_client
        agent._initialized = True

        error = Exception("HTTP Error 429: Too Many Requests")
        context = {"stage": "DOWNLOAD", "extra": "youtube-dl operation"}

        result = agent.classify_error(error, context)

        assert result is not None
        assert result.category == "api"
        assert result.severity == "transient"
        assert "api-healer" in result.suggested_healer

    def test_classify_error_disk_space(self, mock_watcher_config):
        """Test classify_error() handles disk space errors."""
        agent = WatcherAgent(config=mock_watcher_config)

        mock_response = Mock()
        mock_response.parsed_data = {
            "category": "disk",
            "severity": "critical",
            "suggested_healer": "disk-healer",
            "confidence": 0.95,
            "needs_llm_healer": False,
            "reasoning": "Disk full - critical, needs user intervention"
        }

        mock_client = Mock()
        mock_client.generate.return_value = mock_response
        agent.client = mock_client
        agent._initialized = True

        error = OSError("[Errno 28] No space left on device")
        context = {"stage": "DOWNLOAD"}

        result = agent.classify_error(error, context)

        assert result is not None
        assert result.category == "disk"
        assert result.severity == "critical"
        assert "disk-healer" in result.suggested_healer

    def test_classify_error_network_timeout(self, mock_watcher_config):
        """Test classify_error() handles network timeout errors."""
        agent = WatcherAgent(config=mock_watcher_config)

        mock_response = Mock()
        mock_response.parsed_data = {
            "category": "api",
            "severity": "transient",
            "suggested_healer": "api-healer",
            "confidence": 0.88,
            "needs_llm_healer": False,
            "reasoning": "Network timeout - transient, retry should succeed"
        }

        mock_client = Mock()
        mock_client.generate.return_value = mock_response
        agent.client = mock_client
        agent._initialized = True

        import socket
        error = socket.timeout("Connection timed out")
        context = {"stage": "DOWNLOAD"}

        result = agent.classify_error(error, context)

        assert result is not None
        assert result.category == "api"
        assert result.severity == "transient"

    def test_classify_error_returns_none_when_client_not_initialized(self, mock_watcher_config):
        """Test classify_error() returns None when client fails to initialize."""
        agent = WatcherAgent(config=mock_watcher_config)
        agent._initialized = False
        agent.client = None

        # Mock _init_client to do nothing (simulating failure)
        with patch.object(agent, '_init_client', return_value=None):
            error = Exception("Test error")
            context = {"stage": "TEST"}

            result = agent.classify_error(error, context)

            assert result is None

    def test_classify_error_records_fallback_on_failure(self, mock_watcher_config):
        """Test classify_error() records failure with fallback chain."""
        mock_fallback = Mock()
        agent = WatcherAgent(config=mock_watcher_config, fallback_chain=mock_fallback)
        agent._initialized = False
        agent.client = None

        with patch.object(agent, '_init_client', return_value=None):
            error = Exception("Test error")
            context = {"stage": "TEST"}

            result = agent.classify_error(error, context)

            mock_fallback.record_watcher_failure.assert_called_once()

    def test_classify_error_logs_classification_on_success(self, mock_watcher_config):
        """Test classify_error() logs to healing_logger on success."""
        mock_logger = Mock()
        agent = WatcherAgent(config=mock_watcher_config, healing_logger=mock_logger)

        mock_response = Mock()
        mock_response.parsed_data = {
            "category": "api",
            "severity": "transient",
            "suggested_healer": "api-healer",
            "confidence": 0.85,
            "needs_llm_healer": False,
            "reasoning": "Test"
        }

        mock_client = Mock()
        mock_client.generate.return_value = mock_response
        agent.client = mock_client
        agent._initialized = True

        error = Exception("Test error")
        context = {"stage": "TEST"}

        result = agent.classify_error(error, context)

        mock_logger.log_classification.assert_called_once()


class TestWatcherAgentParseResponse:
    """Tests for WatcherAgent._parse_response()."""

    @pytest.fixture
    def agent(self):
        """Create WatcherAgent instance."""
        mock_config = Mock()
        mock_config.escalate_threshold = 0.7
        return WatcherAgent(config=mock_config)

    def test_parse_response_from_parsed_data(self, agent):
        """Test _parse_response() extracts from parsed_data attribute."""
        mock_response = Mock()
        mock_response.parsed_data = {
            "category": "disk",
            "severity": "critical",
            "suggested_healer": "disk-healer",
            "confidence": 0.9,
            "needs_llm_healer": False,
            "reasoning": "Storage full"
        }

        result = agent._parse_response(mock_response, Exception("Test"))

        assert result.category == "disk"
        assert result.severity == "critical"
        assert result.confidence == 0.9

    def test_parse_response_from_text_json(self, agent):
        """Test _parse_response() extracts from text attribute as JSON."""
        mock_response = Mock(spec=['text'])
        mock_response.parsed_data = None
        mock_response.text = '{"category": "path", "severity": "recoverable", "suggested_healer": "path-healer", "confidence": 0.8, "needs_llm_healer": false, "reasoning": "Long path"}'

        result = agent._parse_response(mock_response, Exception("Test"))

        assert result.category == "path"
        assert result.severity == "recoverable"

    def test_parse_response_strips_markdown_code_blocks(self, agent):
        """Test _parse_response() handles markdown code blocks."""
        mock_response = Mock(spec=['text'])
        mock_response.parsed_data = None
        mock_response.text = '```json\n{"category": "api", "severity": "transient", "suggested_healer": "", "confidence": 0.7, "needs_llm_healer": false, "reasoning": "Test"}\n```'

        result = agent._parse_response(mock_response, Exception("Test"))

        assert result.category == "api"
        assert result.severity == "transient"

    def test_parse_response_clamps_confidence(self, agent):
        """Test _parse_response() clamps confidence to [0, 1]."""
        mock_response = Mock()
        mock_response.parsed_data = {
            "category": "unknown",
            "severity": "recoverable",
            "suggested_healer": "",
            "confidence": 1.5,  # Out of range
            "needs_llm_healer": False,
            "reasoning": ""
        }

        result = agent._parse_response(mock_response, Exception("Test"))

        assert result.confidence == 1.0  # Clamped to max

    def test_parse_response_normalizes_invalid_category(self, agent):
        """Test _parse_response() normalizes invalid categories to unknown."""
        mock_response = Mock()
        mock_response.parsed_data = {
            "category": "invalid_category",
            "severity": "recoverable",
            "suggested_healer": "",
            "confidence": 0.5,
            "needs_llm_healer": False,
            "reasoning": ""
        }

        result = agent._parse_response(mock_response, Exception("Test"))

        assert result.category == "unknown"

    def test_parse_response_normalizes_invalid_severity(self, agent):
        """Test _parse_response() normalizes invalid severities to recoverable."""
        mock_response = Mock()
        mock_response.parsed_data = {
            "category": "api",
            "severity": "invalid_severity",
            "suggested_healer": "",
            "confidence": 0.5,
            "needs_llm_healer": False,
            "reasoning": ""
        }

        result = agent._parse_response(mock_response, Exception("Test"))

        assert result.severity == "recoverable"

    def test_parse_response_returns_none_on_invalid_json(self, agent):
        """Test _parse_response() returns None on invalid JSON."""
        mock_response = Mock(spec=['text'])
        mock_response.parsed_data = None
        mock_response.text = "not valid json"

        result = agent._parse_response(mock_response, Exception("Test"))

        assert result is None

    def test_parse_response_defaults_needs_llm_healer_from_confidence(self, agent):
        """Test _parse_response() defaults needs_llm_healer based on confidence."""
        mock_response = Mock()
        mock_response.parsed_data = {
            "category": "unknown",
            "severity": "recoverable",
            "suggested_healer": "",
            "confidence": 0.5  # Below threshold of 0.7
            # needs_llm_healer not provided
        }

        result = agent._parse_response(mock_response, Exception("Test"))

        assert result.needs_llm_healer is True  # Auto-set because confidence < threshold


class TestWatcherAgentShouldEscalate:
    """Tests for WatcherAgent.should_escalate()."""

    @pytest.fixture
    def agent(self):
        """Create WatcherAgent instance."""
        mock_config = Mock()
        mock_config.escalate_threshold = 0.7
        return WatcherAgent(config=mock_config)

    def test_should_escalate_no_classification(self, agent):
        """Test should_escalate() returns True when no classification."""
        assert agent.should_escalate(None) is True

    def test_should_escalate_when_needs_llm_healer(self, agent):
        """Test should_escalate() returns True when needs_llm_healer is True."""
        classification = ErrorClassification(
            category="config",
            severity="critical",
            suggested_healer="",
            confidence=0.9,
            needs_llm_healer=True,
            reasoning=""
        )

        assert agent.should_escalate(classification) is True

    def test_should_escalate_low_confidence(self, agent):
        """Test should_escalate() returns True for low confidence."""
        classification = ErrorClassification(
            category="api",
            severity="transient",
            suggested_healer="api-healer",
            confidence=0.3,  # Below 0.5
            needs_llm_healer=False,
            reasoning=""
        )

        assert agent.should_escalate(classification) is True

    def test_should_not_escalate_high_confidence(self, agent):
        """Test should_escalate() returns False for high confidence, no LLM needed."""
        classification = ErrorClassification(
            category="api",
            severity="transient",
            suggested_healer="api-healer",
            confidence=0.9,
            needs_llm_healer=False,
            reasoning=""
        )

        assert agent.should_escalate(classification) is False

    def test_should_escalate_failed_healer_moderate_confidence(self, agent):
        """Test should_escalate() returns True when healer failed with moderate confidence."""
        classification = ErrorClassification(
            category="disk",
            severity="recoverable",
            suggested_healer="disk-healer",
            confidence=0.75,  # Between 0.5 and 0.8
            needs_llm_healer=False,
            reasoning=""
        )

        failed_healer_result = Mock()
        failed_healer_result.success = False

        assert agent.should_escalate(classification, failed_healer_result) is True

    def test_should_not_escalate_failed_healer_high_confidence(self, agent):
        """Test should_escalate() returns False when healer failed but high confidence."""
        classification = ErrorClassification(
            category="disk",
            severity="recoverable",
            suggested_healer="disk-healer",
            confidence=0.85,  # Above 0.8
            needs_llm_healer=False,
            reasoning=""
        )

        failed_healer_result = Mock()
        failed_healer_result.success = False

        assert agent.should_escalate(classification, failed_healer_result) is False


class TestWatcherAgentGetHealerPriority:
    """Tests for WatcherAgent.get_healer_priority()."""

    @pytest.fixture
    def agent(self):
        """Create WatcherAgent instance."""
        mock_config = Mock()
        return WatcherAgent(config=mock_config)

    def test_get_healer_priority_api_category(self, agent):
        """Test get_healer_priority() returns api-healer for api category."""
        classification = ErrorClassification(
            category="api",
            severity="transient",
            suggested_healer="",
            confidence=0.9,
            needs_llm_healer=False,
            reasoning=""
        )

        result = agent.get_healer_priority(classification)

        assert "api-healer" in result

    def test_get_healer_priority_disk_category(self, agent):
        """Test get_healer_priority() returns disk-healer for disk category."""
        classification = ErrorClassification(
            category="disk",
            severity="critical",
            suggested_healer="",
            confidence=0.9,
            needs_llm_healer=False,
            reasoning=""
        )

        result = agent.get_healer_priority(classification)

        assert "disk-healer" in result

    def test_get_healer_priority_path_category(self, agent):
        """Test get_healer_priority() returns path-healer for path category."""
        classification = ErrorClassification(
            category="path",
            severity="recoverable",
            suggested_healer="",
            confidence=0.9,
            needs_llm_healer=False,
            reasoning=""
        )

        result = agent.get_healer_priority(classification)

        assert "path-healer" in result

    def test_get_healer_priority_suggested_healer_first(self, agent):
        """Test get_healer_priority() puts suggested healer first."""
        classification = ErrorClassification(
            category="api",
            severity="transient",
            suggested_healer="disk-healer",  # Different from category default
            confidence=0.9,
            needs_llm_healer=False,
            reasoning=""
        )

        result = agent.get_healer_priority(classification)

        assert result[0] == "disk-healer"  # Suggested healer first
        assert "api-healer" in result  # Category default still included

    def test_get_healer_priority_unknown_category(self, agent):
        """Test get_healer_priority() returns empty list for unknown category."""
        classification = ErrorClassification(
            category="unknown",
            severity="recoverable",
            suggested_healer="",
            confidence=0.5,
            needs_llm_healer=True,
            reasoning=""
        )

        result = agent.get_healer_priority(classification)

        assert result == []

    def test_get_healer_priority_config_category(self, agent):
        """Test get_healer_priority() returns empty list for config category."""
        classification = ErrorClassification(
            category="config",
            severity="recoverable",
            suggested_healer="",
            confidence=0.5,
            needs_llm_healer=True,
            reasoning=""
        )

        result = agent.get_healer_priority(classification)

        assert result == []


class TestWatcherPromptTemplate:
    """Tests for WATCHER_PROMPT template."""

    def test_prompt_template_has_placeholders(self):
        """Test WATCHER_PROMPT has expected placeholders."""
        assert "{error_message}" in WATCHER_PROMPT
        assert "{error_type}" in WATCHER_PROMPT
        assert "{stage_name}" in WATCHER_PROMPT
        assert "{context}" in WATCHER_PROMPT

    def test_prompt_template_lists_categories(self):
        """Test WATCHER_PROMPT lists all valid categories."""
        categories = ["api", "disk", "path", "checkpoint", "download", "otio", "config", "unknown"]
        for category in categories:
            assert category in WATCHER_PROMPT

    def test_prompt_template_lists_severities(self):
        """Test WATCHER_PROMPT lists all valid severities."""
        assert "critical" in WATCHER_PROMPT
        assert "recoverable" in WATCHER_PROMPT
        assert "transient" in WATCHER_PROMPT

    def test_prompt_template_requests_json(self):
        """Test WATCHER_PROMPT requests JSON output."""
        assert "JSON" in WATCHER_PROMPT
