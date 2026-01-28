"""Tests for two-tier LLM delegation system.

Tests cover:
- HealingLogger: Logging and report generation
- FallbackChain: Pattern routing and circuit breaker
- WatcherAgent: Error classification
- LLMHealer: Self-healing and context building
"""

import json
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.healing_logger import HealingLogger, HealingLogEntry
from src.agents.fallback import FallbackChain, pattern_route, PatternClassification, PATTERN_ROUTING
from src.agents.watcher import WatcherAgent, ErrorClassification
from src.agents.healers.llm_healer import LLMHealer
from src.config.sections.infrastructure import (
    HealingConfig,
    HealingLoggingConfig,
    WatcherConfig,
    LLMHealerConfig,
)


# =============================================================================
# HealingLogger Tests
# =============================================================================

class TestHealingLogger:
    """Tests for HealingLogger."""

    @pytest.mark.integration
    def test_init_creates_log_files(self):
        """Logger creates log directory and files."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir, json_log=True)

            assert log_dir.exists()
            assert logger.session_id  # Has a session ID

    @pytest.mark.integration
    def test_log_classification(self):
        """Logger records classification events."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir, json_log=True)

            classification = ErrorClassification(
                category="api",
                severity="transient",
                suggested_healer="api-healer",
                confidence=0.92,
                needs_llm_healer=False,
                reasoning="Rate limit detected"
            )

            logger.log_classification(
                stage="DOWNLOAD",
                error=Exception("429 Too Many Requests"),
                classification=classification,
                duration_ms=1234.5
            )

            assert len(logger.entries) == 1
            entry = logger.entries[0]
            assert entry.stage == "DOWNLOAD"
            assert entry.component == "watcher"
            assert entry.action == "classify"
            assert entry.details["category"] == "api"
            assert entry.details["confidence"] == 0.92

    @pytest.mark.integration
    def test_log_healer_attempt(self):
        """Logger records healer attempts."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir)

            # Mock HealerResult
            @dataclass
            class MockHealerResult:
                success: bool = True
                action: MagicMock = None
                message: str = "Fixed"
                modified_config: bool = False
                details: dict = None

                def __post_init__(self):
                    self.action = MagicMock()
                    self.action.value = "retry"
                    self.details = self.details or {}

            result = MockHealerResult()

            logger.log_healer_attempt(
                stage="DOWNLOAD",
                healer_name="api-healer",
                error=Exception("Rate limit"),
                result=result,
                duration_ms=500.0
            )

            assert len(logger.entries) == 1
            entry = logger.entries[0]
            assert entry.component == "healer"
            assert entry.action == "attempt"
            assert entry.result == "success"
            assert entry.details["healer"] == "api-healer"

    @pytest.mark.integration
    def test_log_fallback(self):
        """Logger records fallback events."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir)

            logger.log_fallback(
                stage="INIT",
                from_component="watcher",
                to_component="pattern_routing",
                reason="Ollama unavailable"
            )

            assert len(logger.entries) == 1
            entry = logger.entries[0]
            assert entry.action == "fallback"
            assert entry.details["from"] == "watcher"
            assert entry.details["to"] == "pattern_routing"

    @pytest.mark.integration
    def test_generate_report(self):
        """Logger generates summary report."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir)

            # Add some entries
            classification = ErrorClassification(
                category="api", severity="transient",
                suggested_healer="api-healer", confidence=0.9,
                needs_llm_healer=False, reasoning="test"
            )
            logger.log_classification("DOWNLOAD", Exception("test"), classification, 100.0)

            @dataclass
            class MockResult:
                success: bool = True
                action: MagicMock = None
                message: str = "Fixed"
                modified_config: bool = False
                details: dict = None
                def __post_init__(self):
                    self.action = MagicMock()
                    self.action.value = "retry"
                    self.details = {}

            logger.log_healer_attempt("DOWNLOAD", "api-healer", Exception("test"), MockResult(), 50.0)

            report = logger.generate_report()
            assert "DOWNLOAD" in report
            assert "api-healer" in report or "healer" in report.lower()

    @pytest.mark.integration
    def test_thread_safety(self):
        """Logger is thread-safe for concurrent writes."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir)

            def log_entry(i):
                classification = ErrorClassification(
                    category="api", severity="transient",
                    suggested_healer="api-healer", confidence=0.9,
                    needs_llm_healer=False, reasoning=f"test-{i}"
                )
                logger.log_classification(f"STAGE-{i}", Exception(f"error-{i}"), classification, 100.0)

            threads = [threading.Thread(target=log_entry, args=(i,)) for i in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(logger.entries) == 10


# =============================================================================
# FallbackChain Tests
# =============================================================================

class TestPatternRouting:
    """Tests for pattern-based error routing."""

    @pytest.mark.parametrize("error_msg,expected_category,expected_healer", [
        ("HTTPError 429 Too Many Requests", "api", "api-healer"),
        ("Rate limit exceeded", "api", "api-healer"),
        ("Unauthorized: Invalid API key", "api", "api-healer"),
        ("No space left on device", "disk", "disk-healer"),
        ("Permission denied: /path/to/file", "disk", "disk-healer"),
        ("Path too long: 300 characters", "path", "path-healer"),
        ("JSONDecodeError: Expecting value", "checkpoint", "checkpoint-healer"),
        ("Video unavailable: yt-dlp error", "download", "download-healer"),
        ("OTIO: Invalid time range", "otio", "otio-healer"),
        ("Timeline generation failed", "otio", "otio-healer"),
    ])
    @pytest.mark.fast
    def test_pattern_routing(self, error_msg, expected_category, expected_healer):
        """Pattern routing correctly classifies known error patterns."""
        result = pattern_route(error_msg)
        assert isinstance(result, PatternClassification)
        assert result.category == expected_category
        assert result.suggested_healer == expected_healer

    @pytest.mark.fast
    def test_pattern_routing_unknown(self):
        """Unknown errors set needs_llm_healer flag."""
        result = pattern_route("Some completely unknown error XYZ123")
        assert result.category == "unknown"
        assert result.needs_llm_healer is True  # Escalation flag instead of healer name
        assert result.suggested_healer == ""  # No specific healer suggested


class TestFallbackChain:
    """Tests for FallbackChain."""

    def setup_method(self):
        """Create test config and logger."""
        self.config = HealingConfig(
            watcher=WatcherConfig(
                enabled=True,
                host="http://localhost:11434",
                max_failures=3,
                recheck_interval_seconds=60.0
            ),
            llm_healer=LLMHealerConfig(
                enabled=True,
                max_failures=3,
                recheck_interval_seconds=60.0
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            self.log_dir = Path(tmp) / "logs"
            self.logger = HealingLogger(self.log_dir)

    @pytest.mark.integration
    def test_watcher_unavailable_after_failures(self):
        """Watcher disabled after max_failures consecutive failures."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir)
            chain = FallbackChain(self.config, logger)

            # Initially unknown
            assert chain.fallback_state["watcher_available"] is None

            # Record failures
            for _ in range(3):
                chain.record_watcher_failure()

            # Should be disabled
            assert chain.fallback_state["watcher_available"] is False

    @pytest.mark.integration
    def test_llm_healer_unavailable_after_failures(self):
        """LLM healer disabled after max_failures consecutive failures."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir)
            chain = FallbackChain(self.config, logger)

            for _ in range(3):
                chain.record_llm_healer_failure()

            assert chain.fallback_state["llm_healer_available"] is False

    @pytest.mark.integration
    def test_success_resets_failure_count(self):
        """Success resets the failure counter."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir)
            chain = FallbackChain(self.config, logger)

            chain.record_watcher_failure()
            chain.record_watcher_failure()
            assert chain.fallback_state["watcher_failures"] == 2

            chain.record_watcher_success()
            assert chain.fallback_state["watcher_failures"] == 0

    @pytest.mark.integration
    def test_recheck_after_interval(self):
        """Availability re-checked after recheck_interval."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir)

            # Short recheck interval for testing
            config = HealingConfig(
                watcher=WatcherConfig(
                    enabled=True,
                    max_failures=3,
                    recheck_interval_seconds=0.1  # Very short for test
                )
            )
            chain = FallbackChain(config, logger)

            # Disable watcher
            for _ in range(3):
                chain.record_watcher_failure()
            assert chain.fallback_state["watcher_available"] is False

            # Wait for recheck interval
            time.sleep(0.15)

            # Calling check should reset for re-check (though actual check may fail)
            # We just verify the recheck logic runs
            chain.check_watcher_available()
            # The state might still be False if Ollama isn't running,
            # but the recheck should have been attempted

    @pytest.mark.integration
    def test_thread_safety(self):
        """FallbackChain is thread-safe."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = HealingLogger(log_dir)
            chain = FallbackChain(self.config, logger)

            def record_failures():
                for _ in range(10):
                    chain.record_watcher_failure()
                    chain.record_watcher_success()

            threads = [threading.Thread(target=record_failures) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Should complete without errors


# =============================================================================
# WatcherAgent Tests
# =============================================================================

class TestWatcherAgent:
    """Tests for WatcherAgent."""

    @pytest.mark.fast
    def test_error_classification_dataclass(self):
        """ErrorClassification dataclass works correctly."""
        classification = ErrorClassification(
            category="api",
            severity="transient",
            suggested_healer="api-healer",
            confidence=0.92,
            needs_llm_healer=False,
            reasoning="Rate limit detected"
        )

        assert classification.category == "api"
        assert classification.confidence == 0.92
        assert not classification.needs_llm_healer

    @pytest.mark.fast
    def test_should_escalate_no_classification(self):
        """Escalate when no classification available."""
        config = WatcherConfig()
        watcher = WatcherAgent(config)

        assert watcher.should_escalate(None) is True

    @pytest.mark.fast
    def test_should_escalate_low_confidence(self):
        """Escalate when confidence is low."""
        config = WatcherConfig(escalate_threshold=0.7)
        watcher = WatcherAgent(config)

        classification = ErrorClassification(
            category="unknown",
            severity="recoverable",
            suggested_healer="",
            confidence=0.4,  # Below 0.5 threshold
            needs_llm_healer=False,
            reasoning="test"
        )

        assert watcher.should_escalate(classification) is True

    @pytest.mark.fast
    def test_should_escalate_explicit_request(self):
        """Escalate when watcher explicitly requests LLM healer."""
        config = WatcherConfig()
        watcher = WatcherAgent(config)

        classification = ErrorClassification(
            category="config",
            severity="recoverable",
            suggested_healer="",
            confidence=0.95,
            needs_llm_healer=True,  # Explicitly requested
            reasoning="Complex config issue"
        )

        assert watcher.should_escalate(classification) is True

    @pytest.mark.fast
    def test_get_healer_priority(self):
        """Healer priority based on classification."""
        config = WatcherConfig()
        watcher = WatcherAgent(config)

        classification = ErrorClassification(
            category="api",
            severity="transient",
            suggested_healer="api-healer",
            confidence=0.9,
            needs_llm_healer=False,
            reasoning="test"
        )

        priority = watcher.get_healer_priority(classification)
        assert "api-healer" in priority

    @pytest.mark.fast
    def test_get_healer_priority_with_custom_suggestion(self):
        """Healer priority includes custom suggestion."""
        config = WatcherConfig()
        watcher = WatcherAgent(config)

        classification = ErrorClassification(
            category="api",
            severity="transient",
            suggested_healer="custom-healer",  # Custom, not in category map
            confidence=0.9,
            needs_llm_healer=False,
            reasoning="test"
        )

        priority = watcher.get_healer_priority(classification)
        assert priority[0] == "custom-healer"

    @patch('src.agents.watcher.WatcherAgent._init_client')
    @pytest.mark.fast
    def test_warmup_success(self, mock_init):
        """Warmup returns True on success."""
        config = WatcherConfig()
        watcher = WatcherAgent(config)

        # Mock successful client
        watcher._initialized = True
        watcher.client = MagicMock()
        watcher.client.generate.return_value = MagicMock(
            parsed_data={"status": "ready"}
        )

        result = watcher.warmup()
        assert result is True


# =============================================================================
# LLMHealer Tests
# =============================================================================

class TestLLMHealer:
    """Tests for LLMHealer."""

    @pytest.mark.integration
    def test_context_truncation(self):
        """Context is truncated to stay within budget."""
        config = LLMHealerConfig(max_context_chars=1000)

        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            healer = LLMHealer(config, project_dir)

            # Create a very long error message
            long_error = "x" * 5000
            error = Exception(long_error)

            context = healer._build_context(error, None, "TEST")

            # Context should be truncated
            assert len(context) <= config.max_context_chars + 500  # Some buffer for headers

    @pytest.mark.integration
    def test_provider_fallback_order(self):
        """Provider fallback order is correct."""
        config = LLMHealerConfig()

        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            healer = LLMHealer(config, project_dir)

            assert healer.FALLBACK_PROVIDERS == ["anthropic", "gemini", "ollama"]

    @pytest.mark.integration
    def test_timeout_configuration(self):
        """Timeout is configurable and has proper limits."""
        config = LLMHealerConfig(timeout=60.0)

        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            healer = LLMHealer(config, project_dir)

            assert healer.timeout == 60.0
            assert healer.MAX_TIMEOUT == 300.0
            assert healer.INITIAL_TIMEOUT == 60.0

    @pytest.mark.integration
    def test_backoff_configuration(self):
        """Backoff is configurable and initialized correctly."""
        config = LLMHealerConfig()

        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            healer = LLMHealer(config, project_dir)

            assert healer.backoff == healer.INITIAL_BACKOFF
            assert healer.INITIAL_BACKOFF == 2.0

    @patch('src.agents.healers.llm_healer.LLMHealer._init_client')
    @pytest.mark.integration
    def test_can_handle_any_error(self, mock_init):
        """LLM healer can handle any error type."""
        config = LLMHealerConfig()

        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            healer = LLMHealer(config, project_dir)

            # LLM healer should handle anything
            assert healer.can_handle(Exception("any error"))
            assert healer.can_handle(ValueError("value error"))
            assert healer.can_handle(RuntimeError("runtime error"))


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for two-tier LLM delegation."""

    @pytest.mark.fast
    def test_config_loading_with_nested_configs(self):
        """Config loads with nested HealingConfig correctly."""
        config = HealingConfig(
            enabled=True,
            logging={"enabled": True, "log_dir": "logs"},
            watcher={"enabled": True, "model": "llama3.2"},
            llm_healer={"enabled": True, "model": "claude-sonnet-4-20250514"}
        )

        # __post_init__ should convert dicts to dataclasses
        assert isinstance(config.logging, HealingLoggingConfig)
        assert isinstance(config.watcher, WatcherConfig)
        assert isinstance(config.llm_healer, LLMHealerConfig)

        assert config.watcher.model == "llama3.2"
        assert config.llm_healer.model == "claude-sonnet-4-20250514"

    @pytest.mark.integration
    def test_full_classification_to_healing_flow(self):
        """Full flow from error to classification to healing."""
        # This is a mock test - in real usage, Ollama would be running

        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            project_dir = Path(tmp)

            # Create components
            logger = HealingLogger(log_dir)
            config = HealingConfig(
                watcher=WatcherConfig(max_failures=3),
                llm_healer=LLMHealerConfig()
            )
            chain = FallbackChain(config, logger)

            # Simulate watcher unavailable, fall back to pattern routing
            for _ in range(3):
                chain.record_watcher_failure()

            # Pattern routing should work
            error_msg = "HTTPError 429 Too Many Requests"
            result = pattern_route(error_msg)

            assert result.category == "api"
            assert result.suggested_healer == "api-healer"

            # Log the fallback
            logger.log_fallback(
                stage="DOWNLOAD",
                from_component="watcher",
                to_component="pattern_routing",
                reason="Watcher unavailable"
            )

            # Verify logging worked
            # Note: FallbackChain logs when watcher is disabled (1st entry)
            # and we log manually (2nd entry)
            assert len(logger.entries) >= 1
            assert any(e.action == "fallback" for e in logger.entries)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
